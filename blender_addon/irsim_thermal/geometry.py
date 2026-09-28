"""Surface sampling for the connection finder: pure NumPy, no ``bpy``, tested without Blender.

The finder asks two questions of every pair of parts: *how much of A lies against B* (a contact,
conducting through a joint) and *how much of A looks straight at B across a gap* (a facing pair,
exchanging radiation). Both are area questions, and both are answered the same way: scatter points
uniformly **by area** over A's surface and count the ones that satisfy the test. A point stands
for ``total area / n`` of surface, so the estimate is unbiased however the mesh is triangulated.

Counting whole triangles instead would be wrong in exactly the common case: a downloaded shell has
a few large triangles next to many small ones, and one large triangle whose centre happens to lie
under a small bracket would count its whole area as contact.

The facing area is **not a view factor**. It is the area of A whose outward normal, extended up to
a set distance, reaches B before anything else: enough to show a person which parts exchange heat
across a gap, and for them to confirm it. The view factor itself is roadmap row ``TC.9``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Surface:
    """A part's triangles in world space (metres)."""

    vertices: np.ndarray  # (V, 3) float64
    triangles: np.ndarray  # (T, 3) int

    def corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = self.vertices[self.triangles]
        return v[:, 0], v[:, 1], v[:, 2]

    def areas_normals(self) -> tuple[np.ndarray, np.ndarray]:
        """Each triangle's area and unit normal (winding order: counter-clockwise is outward)."""
        a, b, c = self.corners()
        cross = np.cross(b - a, c - a)
        twice = np.linalg.norm(cross, axis=1)
        normals = np.divide(
            cross, twice[:, None], out=np.zeros_like(cross), where=twice[:, None] > 0
        )
        return 0.5 * twice, normals

    @property
    def area(self) -> float:
        return float(self.areas_normals()[0].sum())

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        used = self.vertices[np.unique(self.triangles)] if len(self.triangles) else self.vertices
        return used.min(axis=0), used.max(axis=0)


@dataclass(frozen=True)
class Samples:
    """Points spread uniformly by area over a surface, each standing for ``area_each`` m²."""

    points: np.ndarray  # (n, 3)
    normals: np.ndarray  # (n, 3), the normal of the triangle each point lies on
    area_each: float

    def __len__(self) -> int:
        return len(self.points)


def sample(surface: Surface, n: int, seed: int = 0) -> Samples:
    """``n`` points uniform in area over ``surface`` (a fixed seed, so a rerun finds the same)."""
    areas, normals = surface.areas_normals()
    total = float(areas.sum())
    if total <= 0.0 or n <= 0:
        return Samples(np.zeros((0, 3)), np.zeros((0, 3)), 0.0)
    rng = np.random.default_rng(seed)
    tri = rng.choice(len(areas), size=n, p=areas / total)
    # Uniform in a triangle: fold the unit square's upper half back onto the lower one.
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    a, b, c = surface.corners()
    points = a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])
    return Samples(points, normals[tri], total / n)


def inside_box(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: float) -> np.ndarray:
    """Which points lie in the box ``[lo, hi]`` grown by ``margin`` on every side."""
    return np.all((points >= lo - margin) & (points <= hi + margin), axis=1)


def boxes_overlap(
    lo_a: np.ndarray, hi_a: np.ndarray, lo_b: np.ndarray, hi_b: np.ndarray, margin: float
) -> bool:
    return bool(np.all(lo_a - margin <= hi_b) and np.all(lo_b - margin <= hi_a))


#: Two surfaces touch only where they lie roughly parallel (normals within 60 degrees of parallel
#: or anti-parallel -- either sign, because a downloaded model's normals are not always outward),
#: and only straight across: the nearest point of the other part must lie along the normal, not
#: off to the side. Without the second rule, the strip of a block's top face just outside a box
#: standing on it counts, being within reach of the box's bottom *edge*: at a 2 cm touching
#: distance that is 8 % of a 50 cm box's footprint.
PARALLEL_COS = 0.5


def touching(
    found: np.ndarray,
    normals: np.ndarray,
    other_normals: np.ndarray,
    offsets: np.ndarray,
    touch_m: float,
) -> np.ndarray:
    """Which sample points touch the other part.

    ``found``: a nearest surface point was in reach; ``offsets``: from each sample to it;
    ``other_normals``: the other surface's normal there. A point already on the other surface
    (offset below a thousandth of the touching distance) has no direction to check.
    """
    parallel = np.abs(np.einsum("ij,ij->i", normals, other_normals)) >= PARALLEL_COS
    dist = np.linalg.norm(offsets, axis=1)
    along = np.abs(np.einsum("ij,ij->i", offsets, normals))
    across = (along >= PARALLEL_COS * dist) | (dist <= 1e-3 * touch_m)
    return found & parallel & across


def contact_area(samples: Samples, near: np.ndarray) -> float:
    """The area that the points flagged ``near`` (within the touching distance of B) stand for."""
    return float(np.count_nonzero(near)) * samples.area_each


def default_gaps(largest_m: float) -> tuple[float, float]:
    """``(touching, facing)`` distances for a model whose largest dimension is ``largest_m``.

    Touching: 0.2 % of the model, at least 0.5 mm -- a downloaded model's parts are placed by eye,
    and a motor sitting "on" an arm is often a fraction of a millimetre off it or into it.
    Facing: 10 % of the model, the distance over which one part plausibly warms another by
    radiation (an engine and the bonnet above it, a battery and the shell around it).
    """
    return max(5e-4, 0.002 * largest_m), 0.10 * largest_m
