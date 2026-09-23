"""How much geometry a prepared asset carries, and how much of it a thermal solve can use.

An imported asset arrives with the triangle count a *renderer* wanted. The Phantom 4 of ADR 0128
is 2,486,459 triangles out of a 62 MB FBX, and 1,532,656 after the planar dissolve. Nothing
counted them until this module, so the first import too heavy to solve was discovered by waiting
for it -- which is the worst way to learn it, because a mesh field's cost is linear in cells and a
solve that is going to take an hour looks exactly like one that is going to take a minute for the
first thirty seconds.

There are two separate questions and they have different answers.

**Can we afford it?** `GT.7` measured this: 102,400 cells over a 48-hour spin-up take 11.5 s here,
and Fraunhofer's reference scene was 1,313,410 triangles with a ten-layer stack through five
day-night cycles in 252 s. So 10^6 cells is affordable and 10^5 is comfortable. That fixes
:attr:`GeometryBudget.total_faces`.

**Can we *use* it?** This is the more interesting one, and the answer is usually no. A cell holds
one temperature. Two cells closer together than the distance heat diffuses laterally in one solver
tick cannot hold different temperatures -- conduction erases the difference within the step. That
distance is ``sqrt(alpha * dt)``, and across this project's whole material library it ranges from
**1.1 mm** for the slowest material (`etics_render`, alpha = 2.06e-8 m2/s) to **71 mm** for the
fastest (`bare_aluminium`, 8.44e-5) at a 60-second tick. So a cell finer than about a millimetre
cannot carry an independent temperature for *any* material this project knows about.

Measured on the Phantom 4, one prim carries 100,926 faces over 2.7 cm2 -- a mean cell edge of
**73 µm**, fifteen times below the floor. Those triangles are not a fidelity choice; they are a
cost with no corresponding accuracy, and 69 % of the whole asset's faces are in that category.

The report says both things, and the gate refuses only the affordability one, because "too fine"
is a waste rather than an error and the lever that fixes it -- ``prep_asset.py --dissolve-deg`` --
is the caller's to pull.

docs/physics-model.md §6.6; roadmap `AI.3`; ADR 0128 (the asset pipeline), ADR 0132 (the thermal
mesh is not the render mesh).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, and importing it here would be circular
    from irsim.io.assets import AssetMeshes

__all__ = [
    "FASTEST_DIFFUSIVITY_M2_S",
    "MIN_CELL_EDGE_M",
    "REFERENCE_TICK_S",
    "SLOWEST_DIFFUSIVITY_M2_S",
    "BudgetReport",
    "GeometryBudget",
    "PrimBudget",
    "conduction_length_m",
    "measure_geometry",
]

#: Thermal diffusivity of the slowest material in ``configs/materials/`` (``etics_render``:
#: k = 0.037 W/m/K, rho = 900 kg/m3, c = 2000 J/kg/K). It sets the floor, because no material in
#: the library smears a temperature difference over *less* distance than this one.
SLOWEST_DIFFUSIVITY_M2_S = 2.056e-8
#: And the fastest (``bare_aluminium``: k = 205, rho = 2700, c = 900). Present for the report,
#: which quotes both ends: on a painted aluminium panel, cells closer than 71 mm are already one
#: temperature, so the Phantom 4's shell is over-resolved by three orders of magnitude, not one.
FASTEST_DIFFUSIVITY_M2_S = 8.436e-5
#: The solver tick the floor is quoted at. Shorter ticks resolve finer differences; this is the
#: coarsest of the intervals the drivers in ``scripts/`` actually use, so it is the generous end.
REFERENCE_TICK_S = 60.0


def conduction_length_m(diffusivity_m2_s: float, tick_s: float = REFERENCE_TICK_S) -> float:
    """``sqrt(alpha * dt)`` -- how far a temperature difference spreads laterally in one tick.

    Two cells closer than this hold the same temperature no matter how finely the mesh divides
    them, so it is the resolution beyond which extra geometry buys nothing.
    """
    if diffusivity_m2_s <= 0.0 or tick_s <= 0.0:
        raise ValueError("diffusivity and tick must be positive")
    return math.sqrt(float(diffusivity_m2_s) * float(tick_s))


#: The floor, derived rather than chosen: 1.11 mm.
MIN_CELL_EDGE_M = conduction_length_m(SLOWEST_DIFFUSIVITY_M2_S, REFERENCE_TICK_S)


@dataclass(frozen=True)
class GeometryBudget:
    """What a prepared asset is allowed to carry.

    ``total_faces`` and ``faces_per_prim`` are affordability and are **gated**.
    ``min_cell_edge_m`` is usefulness and is **reported**, because a mesh that is finer than the
    physics can use is wasteful rather than wrong, and the caller may have a reason.
    """

    #: Fraunhofer's measured scene, and the figure `GT.7` sized this lane against.
    total_faces: int = 1_300_000
    #: One prim is one bound surface (ADR 0128 groups by material), and a single prim carrying
    #: more than this is nearly always an imported part that was never meant to be solved.
    faces_per_prim: int = 200_000
    min_cell_edge_m: float = MIN_CELL_EDGE_M

    def __post_init__(self) -> None:
        if self.total_faces <= 0 or self.faces_per_prim <= 0:
            raise ValueError("face budgets must be positive")
        if self.min_cell_edge_m <= 0.0:
            raise ValueError("min_cell_edge_m must be positive")

    def affordable_faces(self, area_m2: float) -> int:
        """How many faces a surface of this area can use at the resolution floor.

        Two triangles per square cell of side ``min_cell_edge_m``, which is the count a regular
        triangulation of that area at that resolution would have.
        """
        if area_m2 <= 0.0:
            return 0
        return max(1, int(math.ceil(2.0 * float(area_m2) / self.min_cell_edge_m**2)))


@dataclass(frozen=True)
class PrimBudget:
    """One prim's geometry, measured."""

    name: str
    material_name: str | None
    faces: int
    area_m2: float
    #: Faces beyond what the conduction length can resolve -- geometry that costs and cannot pay.
    wasted_faces: int = 0

    @property
    def cell_edge_m(self) -> float:
        """Mean cell edge, ``sqrt(2 A / n)``: the side of the square two of these triangles tile.

        Zero for a prim with no area or no faces, which is a degenerate prim rather than an
        infinitely fine one.
        """
        if self.faces <= 0 or self.area_m2 <= 0.0:
            return 0.0
        return math.sqrt(2.0 * self.area_m2 / self.faces)


@dataclass(frozen=True)
class BudgetReport:
    """What :func:`measure_geometry` found, and whether the asset is affordable."""

    asset: str
    budget: GeometryBudget
    prims: tuple[PrimBudget, ...] = field(default_factory=tuple)

    @property
    def total_faces(self) -> int:
        return int(sum(p.faces for p in self.prims))

    @property
    def total_area_m2(self) -> float:
        return float(sum(p.area_m2 for p in self.prims))

    @property
    def wasted_faces(self) -> int:
        return int(sum(p.wasted_faces for p in self.prims))

    @property
    def over_total(self) -> bool:
        return self.total_faces > self.budget.total_faces

    @property
    def over_prim(self) -> tuple[PrimBudget, ...]:
        return tuple(p for p in self.prims if p.faces > self.budget.faces_per_prim)

    @property
    def too_fine(self) -> tuple[PrimBudget, ...]:
        """Prims whose mean cell is finer than any material in the library can resolve.

        Defined as *carrying wasted faces* rather than by comparing the edge to the floor again.
        The two would differ by one triangle at the boundary -- ``affordable_faces`` rounds up, so
        a prim sitting exactly on the budget has an edge a hair below the floor -- and a report
        that says "nothing is wasted" and "this prim is too fine" in the same breath is a report
        nobody trusts the rest of.
        """
        return tuple(p for p in self.prims if p.wasted_faces > 0)

    @property
    def passed(self) -> bool:
        """Affordability only. Being too fine is a waste, and the report says so separately."""
        return not self.over_total and not self.over_prim

    def render(self, top: int = 8) -> str:
        """A report a person reads once and acts on, ordered by what costs most."""
        floor_mm = self.budget.min_cell_edge_m * 1000.0
        lines = [
            f"geometry budget — {self.asset}",
            f"  {len(self.prims)} prims, {self.total_faces:,} faces, {self.total_area_m2:.4f} m2",
            f"  budget: {self.budget.total_faces:,} faces total, "
            f"{self.budget.faces_per_prim:,} per prim (GT.7)",
            f"  resolution floor: {floor_mm:.2f} mm — heat crosses that in one "
            f"{REFERENCE_TICK_S:.0f} s tick in the slowest material in the library, so finer "
            f"cells cannot hold different temperatures",
            "",
            f"  {'faces':>10}  {'area m2':>9}  {'cell':>9}  {'waste':>6}  prim",
        ]
        for p in sorted(self.prims, key=lambda q: (-q.faces, q.name))[:top]:
            edge = f"{p.cell_edge_m * 1000.0:.3f}mm" if p.cell_edge_m else "—"
            waste = f"{100.0 * p.wasted_faces / p.faces:.0f}%" if p.faces else "—"
            lines.append(f"  {p.faces:>10,}  {p.area_m2:>9.5f}  {edge:>9}  {waste:>6}  {p.name}")
        if len(self.prims) > top:
            lines.append(f"  ... {len(self.prims) - top} more")
        lines.append("")
        if self.wasted_faces:
            share = 100.0 * self.wasted_faces / max(self.total_faces, 1)
            lines.append(
                f"  {self.wasted_faces:,} faces ({share:.0f} %) are finer than the floor: they "
                "cost a solve and cannot carry a gradient."
            )
        if self.over_total:
            lines.append(
                f"  REFUSED: {self.total_faces:,} faces is over the {self.budget.total_faces:,} "
                "budget. Raise --dissolve-deg, or pass --allow-over-budget and accept the solve "
                "time."
            )
        for p in self.over_prim:
            lines.append(
                f"  REFUSED: {p.name} carries {p.faces:,} faces, over the "
                f"{self.budget.faces_per_prim:,} per-prim budget."
            )
        if self.passed:
            lines.append("  within budget.")
        return "\n".join(lines)


def measure_geometry(
    meshes: AssetMeshes,
    budget: GeometryBudget | None = None,
    *,
    asset: str | None = None,
) -> BudgetReport:
    """Measure a prepared asset's geometry against a budget. Engine-free: reads the ``.npz``."""
    budget = budget or GeometryBudget()
    prims: list[PrimBudget] = []
    for name, mesh in sorted(meshes.meshes.items()):
        faces = int(mesh.n_faces)
        area = float(mesh.area_m2)
        affordable = budget.affordable_faces(area)
        prims.append(
            PrimBudget(
                name=name,
                material_name=mesh.material_name,
                faces=faces,
                area_m2=area,
                wasted_faces=max(faces - affordable, 0) if area > 0.0 else 0,
            )
        )
    return BudgetReport(asset=asset or meshes.name, budget=budget, prims=tuple(prims))
