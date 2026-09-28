"""How much of the model has an infrared material, and what is still missing.

Pure Python + NumPy, no ``bpy``: the add-on hands this module per-face arrays that Blender reads
with ``foreach_get``, so the arithmetic is testable without Blender.

Coverage is **area-weighted**, the same way ``irsim.materials.mapping.audit`` weighs prims against
ADR 0047's 95 % gate: a missing rivet is not a missing wing. The add-on asks for 100 % because
every face it can see is one a person can click, but it reports the gate the pipeline will apply.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "COVERAGE_GATE",
    "MIRROR_EMISSIVITY",
    "PartStats",
    "Summary",
    "part_stats",
    "summarize",
]

#: ADR 0047: the fraction of area `scripts/prep_asset.py` requires mapped before it passes an asset.
COVERAGE_GATE = 0.95

#: `irsim.materials.mapping.MIRROR_EMISSIVITY`. Below this LWIR emissivity a surface shows mostly
#: reflected surroundings, not its own temperature -- legitimate for polished metal, and a classic
#: mistake for a housing that is really painted or anodised (the Phantom 4's motors, ADR 0128).
MIRROR_EMISSIVITY = 0.2


@dataclass
class PartStats:
    """One part (a Blender mesh object): its area, and how much of it has which thermal material."""

    name: str
    total_area_m2: float
    by_thermal_m2: dict[str, float] = field(default_factory=dict)
    #: Blender materials on this part's faces that carry no thermal material, with their area.
    unassigned_by_material_m2: dict[str, float] = field(default_factory=dict)

    @property
    def assigned_area_m2(self) -> float:
        return float(sum(self.by_thermal_m2.values()))

    @property
    def unassigned_area_m2(self) -> float:
        return max(0.0, self.total_area_m2 - self.assigned_area_m2)

    @property
    def fully_assigned(self) -> bool:
        return self.total_area_m2 > 0.0 and self.unassigned_area_m2 <= 1e-12 * self.total_area_m2


@dataclass
class Summary:
    total_area_m2: float
    assigned_area_m2: float
    parts: list[PartStats]
    #: ``(part, thermal material, LWIR emissivity)`` for every assignment below the mirror limit.
    mirror_parts: list[tuple[str, str, float]]

    @property
    def coverage(self) -> float:
        return 0.0 if self.total_area_m2 <= 0.0 else self.assigned_area_m2 / self.total_area_m2

    @property
    def unassigned_parts(self) -> list[PartStats]:
        """Parts with any face lacking a thermal material, largest missing area first."""
        missing = [p for p in self.parts if not p.fully_assigned and p.total_area_m2 > 0.0]
        return sorted(missing, key=lambda p: -p.unassigned_area_m2)

    @property
    def passes_gate(self) -> bool:
        return self.coverage >= COVERAGE_GATE


def part_stats(
    name: str,
    face_areas_m2: NDArray[np.floating],
    face_slots: NDArray[np.integer],
    slot_materials: Sequence[str | None],
    slot_thermals: Sequence[str | None],
) -> PartStats:
    """Area per thermal material for one part.

    ``face_slots[i]`` is face *i*'s material slot; ``slot_materials[s]`` is the Blender material in
    slot *s* (None for an empty slot) and ``slot_thermals[s]`` its thermal material (None or ``""``
    when unassigned). A face whose slot index is past the end of the slot list -- Blender allows
    that and renders it with no material -- counts as unassigned, under the name ``<no material>``.
    """
    areas = np.asarray(face_areas_m2, dtype=np.float64)
    slots = np.asarray(face_slots, dtype=np.int64)
    if areas.shape != slots.shape:
        raise ValueError("one area and one slot index per face")
    stats = PartStats(name=name, total_area_m2=float(areas.sum()))
    n_slots = len(slot_materials)
    if areas.size == 0:
        return stats
    per_slot = np.bincount(np.clip(slots, 0, n_slots), weights=areas, minlength=n_slots + 1)
    for s in range(n_slots + 1):
        area = float(per_slot[s])
        if area <= 0.0:
            continue
        material = slot_materials[s] if s < n_slots else None
        thermal = slot_thermals[s] if s < n_slots else None
        if thermal:
            stats.by_thermal_m2[thermal] = stats.by_thermal_m2.get(thermal, 0.0) + area
        else:
            key = material or "<no material>"
            stats.unassigned_by_material_m2[key] = (
                stats.unassigned_by_material_m2.get(key, 0.0) + area
            )
    return stats


def summarize(
    parts: Sequence[PartStats],
    emissivity_lwir: Mapping[str, float],
    mirror_limit: float = MIRROR_EMISSIVITY,
) -> Summary:
    total = float(sum(p.total_area_m2 for p in parts))
    assigned = float(sum(p.assigned_area_m2 for p in parts))
    mirrors = [
        (p.name, thermal, float(emissivity_lwir[thermal]))
        for p in parts
        for thermal in sorted(p.by_thermal_m2)
        if thermal in emissivity_lwir and emissivity_lwir[thermal] < mirror_limit
    ]
    return Summary(total, assigned, list(parts), mirrors)
