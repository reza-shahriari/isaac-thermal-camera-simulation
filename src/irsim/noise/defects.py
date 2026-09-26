"""Bad-pixel map and defect injection (M9.5a).

docs/physics-model.md §10.4, §11.1. ADR 0055.

§10.4 asks for four defect classes — dead, hot, flickering, blinking — at 0.05–0.5 % of pixels,
"clustered slightly (use a Poisson cluster process, not uniform)". Clustering is not decoration:
defects come from wafer and readout faults that are spatially correlated, so real maps have pairs
and triples where a uniform map has none. A cluster is what defeats the 4-neighbour replacement of
M9.5b, so a simulator that scatters defects uniformly systematically *understates* the artefact
the perception stack actually sees.

**The process (ADR 0055): Neyman–Scott.** Parent points land uniformly at random; each parent
carries ``1 + Poisson(λ)`` offspring placed uniformly in a disc of radius ``CLUSTER_RADIUS_PX``
around it, with λ = ``noise.bad_pixel_cluster_lambda``. The parent count is set so the expected
offspring total matches ``noise.bad_pixel_fraction`` of the array. Offspring that collide on one
pixel are one defect, so the realised count sits slightly below the target — bounded and measured,
not ignored (see :func:`generate_map`).

**The four classes.** The first two are static and the second two carry per-frame state:

* ``DEAD`` — stuck at the DN floor. Unresponsive, every frame, forever.
* ``HOT`` — stuck at the DN ceiling, likewise.
* ``FLICKERING`` — random telegraph noise. The pixel still *responds*; its offset hops between two
  levels by a two-state Markov chain, so the dwell in each state is geometric. This is the class
  that survives a bad-pixel map built from a single calibration frame, because half the time it
  looks fine.
* ``BLINKING`` — intermittent failure. The same two-state chain, but in its bad state the pixel is
  stuck like a dead one rather than merely offset.

**Where in the chain.** Injection is on the **raw DN plane**, matching §11.1's
``raw DN → bad-pixel replace``. A stuck pixel is physically stuck before the ADC, but the ADC is
monotone and clipping, so stuck-low in signal space and DN 0 are the same observable; working in DN
makes "bit-identical across 100 frames" exact rather than approximate, and puts the defect and its
M9.5b repair on the same plane.

The map is drawn once per sensor from the ``BAD_PIXEL_MAP`` stream and the per-frame states from
``RTS`` (ADR 0022), so a camera's defects are a property of the camera, not of the scene.

**Factory and late defects (SC.19, §10.4).** A camera replaces the pixels on the defect map it was
shipped with, not the pixels that are actually bad. ``noise.bad_pixel_late_fraction`` of the
population is marked *late* -- it failed after the map was made -- and :func:`replacement_mask`
leaves it out, so a late hot pixel reaches the 8-bit output as an isolated full-scale dot. The late
flags are drawn from the same stream **after** every other draw, so a map's positions and classes
are bit-identical whatever the fraction is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import BadPixelTypeMix, NoiseSpec
from irsim.noise.seeding import NoiseStream, noise_rng, sensor_rng

__all__ = [
    "DefectKind",
    "BadPixelMap",
    "DefectState",
    "CLUSTER_RADIUS_PX",
    "generate_map",
    "apply_defects",
    "active_defect_mask",
    "replacement_mask",
]

# ADR 0055: offspring land within two pixels of their parent. Big enough to make 2x2 and small
# 3x3 clusters -- the ones that defeat 4-neighbour replacement -- and small enough that a cluster
# stays a cluster rather than a diffuse sprinkle.
CLUSTER_RADIUS_PX = 2.0


class DefectKind(IntEnum):
    """Defect class per pixel; ``GOOD`` is 0 so the plane doubles as a mask."""

    GOOD = 0
    DEAD = 1
    HOT = 2
    FLICKERING = 3
    BLINKING = 4


#: The classes whose behaviour changes frame to frame, and so need persistent state.
STATEFUL_KINDS = (DefectKind.FLICKERING, DefectKind.BLINKING)


@dataclass(frozen=True)
class BadPixelMap:
    """Which pixels are bad and how (§10.4). Drawn once per sensor; never per frame."""

    kind: NDArray[np.uint8]  # DefectKind per pixel, shape (rows, cols)
    #: True where the defect appeared after the factory map was made (SC.19); None = none did.
    late: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        k = np.asarray(self.kind)
        if k.ndim != 2 or k.dtype != np.uint8:
            raise ValueError(f"kind must be a 2-D uint8 plane, got {k.ndim}-D {k.dtype}")
        if k.max(initial=0) > int(DefectKind.BLINKING):
            raise ValueError("kind contains a value outside DefectKind")
        if self.late is not None:
            late = np.asarray(self.late)
            if late.shape != k.shape or late.dtype != np.bool_:
                raise ValueError("late must be a bool plane of the map's shape")
            if np.any(late & (k == int(DefectKind.GOOD))):
                raise ValueError("a good pixel cannot be a late defect")

    @property
    def late_mask(self) -> NDArray[np.bool_]:
        """True where the defect is not on the camera's factory map (SC.19)."""
        if self.late is None:
            return np.zeros(self.shape, dtype=bool)
        return np.asarray(self.late)

    @property
    def factory_mask(self) -> NDArray[np.bool_]:
        """The camera's own defect map: every defect except the late ones."""
        return np.asarray(self.mask & ~self.late_mask)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.kind.shape[0]), int(self.kind.shape[1]))

    @property
    def mask(self) -> NDArray[np.bool_]:
        """True where the pixel is defective in any class."""
        return np.asarray(self.kind != int(DefectKind.GOOD))

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.mask))

    def mask_of(self, kind: DefectKind) -> NDArray[np.bool_]:
        return np.asarray(self.kind == int(kind))

    def counts(self) -> dict[DefectKind, int]:
        return {k: int(np.count_nonzero(self.mask_of(k))) for k in DefectKind}

    @property
    def stateful_mask(self) -> NDArray[np.bool_]:
        """True where the pixel's behaviour changes frame to frame (flickering or blinking)."""
        out = np.zeros(self.shape, dtype=bool)
        for kind in STATEFUL_KINDS:
            out |= self.mask_of(kind)
        return out

    @property
    def stateful_index(self) -> NDArray[np.int64]:
        """Flat indices of the stateful pixels.

        The Markov chain is advanced only over these. A 640x512 focal plane with a few hundred
        defects would otherwise draw 327k uniforms per frame to move a few dozen bits, which at
        60 fps is most of the noise chain's cost spent on nothing.
        """
        return np.asarray(np.flatnonzero(self.stateful_mask), dtype=np.int64)


@dataclass
class DefectState:
    """Per-frame state of the flickering and blinking pixels (§10.4).

    ``bad`` is True where a stateful defect is currently in its bad state. It is carried across
    frames: a random telegraph signal whose state was redrawn independently every frame would have
    white dwell statistics instead of geometric ones, and would look like ordinary per-pixel noise
    rather than the slow winking that makes RTS recognisable.
    """

    bad: NDArray[np.bool_]

    @classmethod
    def initial(cls, bad_map: BadPixelMap, noise: NoiseSpec, sensor_seed: int) -> DefectState:
        """Draw the starting state from the stationary distribution, so there is no burn-in."""
        rng = noise_rng(sensor_seed, 0, NoiseStream.RTS)
        index = bad_map.stateful_index
        bad = np.zeros(bad_map.shape, dtype=bool)
        if index.size:
            draw = rng.random(index.size) < float(noise.bad_pixel_rts_occupancy)
            bad.reshape(-1)[index] = draw
        return cls(bad=bad)


def _parent_count(shape: tuple[int, int], noise: NoiseSpec) -> int:
    """Parents needed so that E[offspring] equals the configured defective fraction."""
    target = float(noise.bad_pixel_fraction) * shape[0] * shape[1]
    per_parent = 1.0 + float(noise.bad_pixel_cluster_lambda)
    return int(round(target / per_parent))


def generate_map(shape: tuple[int, int], noise: NoiseSpec, sensor_seed: int) -> BadPixelMap:
    """Draw one sensor's defect map by the §10.4 Neyman–Scott cluster process (ADR 0055).

    Parents are uniform over the array; each carries ``1 + Poisson(λ)`` offspring inside a disc of
    ``CLUSTER_RADIUS_PX``. Offspring landing outside the array are dropped and colliding offspring
    merge, so the realised count runs a few per cent under ``bad_pixel_fraction`` — the loss is the
    price of clustering and is measured by the tests rather than compensated for, because inflating
    the parent count to hit an exact total would change the cluster-size distribution, which is the
    part that actually matters downstream.

    Deterministic in ``sensor_seed``: a camera's defects belong to the camera.
    """
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"shape must be positive, got {shape}")
    kind = np.zeros((rows, cols), dtype=np.uint8)
    if noise.bad_pixel_fraction <= 0.0:
        return BadPixelMap(kind=kind)

    rng = sensor_rng(sensor_seed, NoiseStream.BAD_PIXEL_MAP)
    n_parents = _parent_count((rows, cols), noise)
    if n_parents == 0:
        return BadPixelMap(kind=kind)

    py = rng.uniform(0.0, rows, size=n_parents)
    px = rng.uniform(0.0, cols, size=n_parents)
    n_offspring = 1 + rng.poisson(float(noise.bad_pixel_cluster_lambda), size=n_parents)
    total = int(n_offspring.sum())

    # Uniform in the disc: sqrt of a uniform radius, else offspring pile up at the centre.
    theta = rng.uniform(0.0, 2.0 * np.pi, size=total)
    radius = CLUSTER_RADIUS_PX * np.sqrt(rng.uniform(0.0, 1.0, size=total))
    cy = np.repeat(py, n_offspring)
    cx = np.repeat(px, n_offspring)
    yy = np.floor(cy + radius * np.sin(theta)).astype(np.int64)
    xx = np.floor(cx + radius * np.cos(theta)).astype(np.int64)

    inside = (yy >= 0) & (yy < rows) & (xx >= 0) & (xx < cols)
    yy, xx = yy[inside], xx[inside]

    # Merge collisions, then assign classes to the surviving distinct pixels.
    flat = np.unique(yy * cols + xx)
    if flat.size == 0:
        return BadPixelMap(kind=kind)
    classes = _draw_classes(flat.size, noise.bad_pixel_type_mix, rng)
    kind.reshape(-1)[flat] = classes
    late = None
    if noise.bad_pixel_late_fraction > 0.0:
        # Drawn last, so positions and classes do not depend on the fraction (SC.19).
        late = np.zeros((rows, cols), dtype=bool)
        late.reshape(-1)[flat] = rng.random(flat.size) < float(noise.bad_pixel_late_fraction)
    return BadPixelMap(kind=kind, late=late)


def _draw_classes(n: int, mix: BadPixelTypeMix, rng: np.random.Generator) -> NDArray[np.uint8]:
    """Assign each defect a §10.4 class by the configured mix, independently of its cluster.

    Independent because the mix describes a population and nothing in §10.4 says a cluster shares
    a failure mode; a mixed-class cluster is also the harder case for M9.5b's replacement.
    """
    probs = np.array([mix.dead, mix.hot, mix.flickering, mix.blinking], dtype=np.float64)
    probs = probs / probs.sum()
    kinds = np.array(
        [DefectKind.DEAD, DefectKind.HOT, DefectKind.FLICKERING, DefectKind.BLINKING],
        dtype=np.uint8,
    )
    return kinds[rng.choice(len(kinds), size=n, p=probs)]


def advance_state(
    state: DefectState,
    bad_map: BadPixelMap,
    noise: NoiseSpec,
    sensor_seed: int,
    frame_index: int,
) -> DefectState:
    """Step the two-state Markov chain of the flickering and blinking pixels by one frame.

    Switch probabilities are chosen so that the stationary occupancy of the bad state is
    ``bad_pixel_rts_occupancy`` and the mean dwell in it is ``bad_pixel_rts_dwell_frames``. Both
    dwell times are then geometric, which is the defining statistic of random telegraph noise and
    what distinguishes it from a pixel that is simply noisy.
    """
    occupancy = float(noise.bad_pixel_rts_occupancy)
    dwell = float(noise.bad_pixel_rts_dwell_frames)
    p_leave_bad = 1.0 / dwell
    # Detailed balance: occ * p_leave_bad = (1 - occ) * p_enter_bad
    p_enter_bad = p_leave_bad * occupancy / (1.0 - occupancy)
    if not 0.0 <= p_enter_bad <= 1.0:
        raise ValueError(
            f"bad_pixel_rts_occupancy {occupancy} with dwell {dwell} frames implies a switch "
            f"probability of {p_enter_bad:.3f}; lower the occupancy or raise the dwell"
        )

    index = bad_map.stateful_index
    bad = np.zeros(bad_map.shape, dtype=bool)
    if index.size:
        rng = noise_rng(sensor_seed, frame_index, NoiseStream.RTS)
        u = rng.random(index.size)
        was_bad = state.bad.reshape(-1)[index]
        bad.reshape(-1)[index] = np.where(was_bad, u >= p_leave_bad, u < p_enter_bad)
    return DefectState(bad=bad)


def active_defect_mask(bad_map: BadPixelMap, state: DefectState) -> NDArray[np.bool_]:
    """The pixels that are actually defective *this frame* — the mask M9.5b replaces.

    Dead and hot pixels are always in it; a blinking or flickering one only while its state is
    bad. This is the truth; what the camera replaces is :func:`replacement_mask`.
    """
    static = bad_map.mask_of(DefectKind.DEAD) | bad_map.mask_of(DefectKind.HOT)
    return np.asarray(static | (bad_map.stateful_mask & state.bad))


def replacement_mask(bad_map: BadPixelMap, state: DefectState) -> NDArray[np.bool_]:
    """What the camera interpolates over this frame: the active defects on its factory map.

    §10.4 (SC.19). A late defect is active but not on the map, so it is left in the image. An
    intermittent defect on the map is still replaced only while bad -- a static map cannot know
    when a flickering pixel is in its good state, but interpolating a good pixel is invisible, so
    replacing only while bad is the same image.
    """
    return np.asarray(active_defect_mask(bad_map, state) & bad_map.factory_mask)


def apply_defects(
    dn: NDArray[np.unsignedinteger],
    bad_map: BadPixelMap,
    state: DefectState,
    dn_max: int,
    rts_amplitude_dn: float,
) -> NDArray[np.unsignedinteger]:
    """Inject the defects into a raw DN plane (§11.1 ``raw DN → bad-pixel replace``).

    Dead and hot pixels are pinned to the floor and the ceiling every frame — bit-identical
    forever, which is what makes them findable by a calibration. A blinking pixel is pinned only
    while its state is bad. A flickering pixel is *offset* by ``rts_amplitude_dn`` while bad and
    otherwise untouched: it keeps responding to the scene, which is exactly why a map built from
    one calibration frame misses it.

    The input is not modified.
    """
    arr = np.asarray(dn)
    if arr.shape != bad_map.shape:
        raise ValueError(f"frame shape {arr.shape} != bad-pixel map shape {bad_map.shape}")
    if state.bad.shape != bad_map.shape:
        raise ValueError("defect state shape does not match the map")
    if not np.issubdtype(arr.dtype, np.unsignedinteger):
        raise TypeError(f"defects are injected on the raw DN plane, got {arr.dtype}")

    out = np.array(arr, copy=True)
    out[bad_map.mask_of(DefectKind.DEAD)] = 0
    out[bad_map.mask_of(DefectKind.HOT)] = dn_max
    out[bad_map.mask_of(DefectKind.BLINKING) & state.bad] = 0

    flicker = bad_map.mask_of(DefectKind.FLICKERING) & state.bad
    if np.any(flicker):
        shifted = out[flicker].astype(np.int64) + int(round(rts_amplitude_dn))
        out[flicker] = np.clip(shifted, 0, dn_max).astype(out.dtype)
    return out
