"""A surface temperature that varies **across** one surface, not one value per object.

docs/physics-model.md §6.1 (the balance), §6.4 (the fixed tick); ADR 0087.

The spec solves a temperature per *surface* and the code has followed it: `ThermalField` carries
N facets, but every consumer so far has mapped one facet to one prim, so a rendered object is one
temperature everywhere. That is wrong in the way that matters most in LWIR, where the temperature
field *is* the image: a car bonnet with the engine running is 15 K hotter over the block than at
its corners, and the asphalt under it is warmer than the asphalt a metre away. Rendering either as
a flat patch removes the only structure a detector would key on.

This module is the smallest honest fix. A :class:`PlanarPatch` is a rectangular grid of cells on a
plane; its cells are ordinary §6.1 facets, so they are solved by the `FacetSolver` that M6.9
already holds to 0.1 mK against N scalar runs, and the spatial part is *only* the lookup --
a point in space becomes a cell, and the query is bilinear so the answer has no cell staircase in
it. Nothing about the balance changes. What changes is that `FacetForcing.q_internal_w_m2`,
`q_solar_w_m2` and the rest may now differ from cell to cell across one piece of geometry, which
is what a heat source under a panel and a shadow across a road actually do.

**Why a projected plane and not a mesh or a UV atlas.** The renderer must be able to say which
cell a pixel landed on. ADR 0014 measured what this build transports per pixel: an exact integer
instance id, and `Camera3dPositionSD`/`PtWorldPos` as a **float32 world position good to 3.4 mm**.
There is no UV AOV and no per-triangle id, and every colour AOV is fp16. A world position is
therefore the only per-pixel surface parameterisation available, and projecting it onto a plane is
the one mapping that needs nothing else -- no atlas to author, no unwrap to keep in sync with the
mesh, no new renderer capability to wait on. The cost is that it is exact only for near-planar
surfaces (a road, a bonnet, a roof, a deck). A cylinder or a wheel still takes one temperature per
prim, and ADR 0087 records that as the known limit rather than hiding it.

**The trap: a rectangle is not a surface.** A patch over the asphalt and a bonnet 0.9 m above it
project into the *same* (u, v) rectangle, so a patch that tested only its in-plane extent would
hand the road's temperature to the car's bonnet, and the picture would look entirely reasonable.
Every patch therefore claims a **slab**, not a rectangle: :attr:`PlanarPatch.thickness_m` is the
half-width normal to the plane, and a point outside it is outside the patch. `test_surface_field`
pins this with a bonnet directly above a road patch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.field import DEFAULT_TICK_S, ThermalField

__all__ = ["PlanarPatch", "PlanarThermalField", "DEFAULT_KEEP_TICKS"]

#: The bracketing pair a query blends, and nothing older: what a spatial field holds by default.
DEFAULT_KEEP_TICKS = 2


def _unit(vector: Any, what: str) -> NDArray[np.float64]:
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    if v.shape != (3,):
        raise ValueError(f"{what} must be a 3-vector, got shape {np.shape(vector)}")
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        raise ValueError(f"{what} has zero length")
    return np.asarray(v / norm)


def _points(value: Any) -> NDArray[np.float64]:
    p = np.asarray(value, dtype=np.float64)
    if p.ndim == 0 or p.shape[-1] != 3:
        raise ValueError(f"points must have a trailing axis of 3, got shape {p.shape}")
    return p


@dataclass(frozen=True)
class PlanarPatch:
    """A rectangular grid of thermal cells on a plane, in one named frame.

    Cell ``(i_v, i_u)`` is centred at ``origin_m + (i_u + 1/2) du u + (i_v + 1/2) dv v``. Cells
    are flattened in C order with ``v`` as the slow axis, so a flat ``(n_cells,)`` temperature
    array reshapes to ``(n_v, n_u)`` and reads like an image of the surface.

    ``frame`` is a name, not a transform: ``"world"`` for a road, a prim path for a panel that
    moves with its object. The caller is responsible for putting query points in the same frame,
    and :class:`~irsim_isaac.pipeline.point_bridge` checks the name rather than assuming it.
    """

    origin_m: NDArray[np.float64]
    u_axis: NDArray[np.float64]
    v_axis: NDArray[np.float64]
    n_u: int
    n_v: int
    du_m: float
    dv_m: float
    thickness_m: float = 0.25
    frame: str = "world"

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_m", np.asarray(self.origin_m, dtype=np.float64).reshape(3))
        u = _unit(self.u_axis, "u_axis")
        v = _unit(self.v_axis, "v_axis")
        # Orthogonality is required rather than silently fixed by Gram-Schmidt: a caller who
        # passed two axes 5 degrees from perpendicular meant something, and a quietly squared-up
        # grid would sample a surface the caller never described.
        if abs(float(np.dot(u, v))) > 1e-9:
            raise ValueError("u_axis and v_axis must be perpendicular")
        object.__setattr__(self, "u_axis", u)
        object.__setattr__(self, "v_axis", v)
        if self.n_u < 1 or self.n_v < 1:
            raise ValueError("a patch needs at least one cell in each direction")
        if self.du_m <= 0.0 or self.dv_m <= 0.0:
            raise ValueError("cell spacings must be positive")
        if self.thickness_m <= 0.0:
            raise ValueError("thickness_m must be positive: a patch claims a slab, not a plane")

    # -- geometry ---------------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return int(self.n_u) * int(self.n_v)

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_v, n_u)`` -- the shape a flat cell array reshapes to."""
        return (int(self.n_v), int(self.n_u))

    @property
    def normal(self) -> NDArray[np.float64]:
        return np.asarray(np.cross(self.u_axis, self.v_axis))

    @property
    def extent_m(self) -> tuple[float, float]:
        return (self.n_u * self.du_m, self.n_v * self.dv_m)

    @property
    def cell_area_m2(self) -> float:
        return float(self.du_m * self.dv_m)

    @property
    def area_m2(self) -> float:
        """The whole patch, ``n_cells · A_cell`` -- a panel's area for a lumped coupling."""
        return self.cell_area_m2 * self.n_cells

    def cell_centres(self) -> NDArray[np.float64]:
        """``(n_cells, 3)`` cell-centre positions, C order with ``v`` slow."""
        iu = (np.arange(self.n_u, dtype=np.float64) + 0.5) * self.du_m
        iv = (np.arange(self.n_v, dtype=np.float64) + 0.5) * self.dv_m
        grid_v, grid_u = np.meshgrid(iv, iu, indexing="ij")
        return np.asarray(
            self.origin_m
            + grid_u.reshape(-1, 1) * self.u_axis
            + grid_v.reshape(-1, 1) * self.v_axis
        )

    def local_coords(self, points: Any) -> NDArray[np.float64]:
        """``(..., 3)`` points -> ``(..., 3)`` of (u, v, w) metres, w normal to the plane."""
        p = _points(points) - self.origin_m
        return np.stack(
            [p @ self.u_axis, p @ self.v_axis, p @ self.normal],
            axis=-1,
        )

    def contains(self, points: Any) -> NDArray[np.bool_]:
        """Inside the patch's **slab**: in the rectangle *and* within ``thickness_m`` of it."""
        return self.contains_local(self.local_coords(points))

    def contains_local(self, uvw: Any) -> NDArray[np.bool_]:
        """:meth:`contains` for points already in patch coordinates.

        Split out for `GT.7`. :meth:`sample` needs both the coordinates and the inside test, and
        computing the coordinates is the expensive half -- three dot products over every pixel of
        the frame. Calling :meth:`contains` from :meth:`sample` computed them a second time and
        threw the first copy away, which measured **27 %** of the sampling cost at 640x512.
        """
        local = np.asarray(uvw, dtype=np.float64)
        span_u, span_v = self.extent_m
        return np.asarray(
            (local[..., 0] >= 0.0)
            & (local[..., 0] <= span_u)
            & (local[..., 1] >= 0.0)
            & (local[..., 1] <= span_v)
            & (np.abs(local[..., 2]) <= self.thickness_m)
        )

    # -- the lookup -------------------------------------------------------------------------

    def sample(
        self, values: Any, points: Any, *, fill: float = float("nan")
    ) -> NDArray[np.float64]:
        """Bilinear sample of a ``(n_cells,)`` cell array at arbitrary points.

        Exact at cell centres and linear between them, so the surface carries no cell staircase
        at the resolutions a camera resolves. Points outside the slab get ``fill``, which is NaN
        by default *because* NaN propagates: a patch silently returning its edge value for
        everything behind it is the failure this fill exists to make loud.
        """
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size != self.n_cells:
            raise ValueError(f"values has {flat.size} cells, patch has {self.n_cells}")
        grid = flat.reshape(self.shape)

        uvw = self.local_coords(points)
        inside = self.contains_local(uvw)  # GT.7: not `contains(points)`, which recomputes uvw
        # Continuous cell coordinate: 0.0 at the centre of cell 0, 1.0 at the centre of cell 1.
        cu = np.clip(uvw[..., 0] / self.du_m - 0.5, 0.0, self.n_u - 1.0)
        cv = np.clip(uvw[..., 1] / self.dv_m - 0.5, 0.0, self.n_v - 1.0)
        u0 = np.floor(cu).astype(np.intp)
        v0 = np.floor(cv).astype(np.intp)
        u1 = np.minimum(u0 + 1, self.n_u - 1)
        v1 = np.minimum(v0 + 1, self.n_v - 1)
        fu = cu - u0
        fv = cv - v0
        out = (
            grid[v0, u0] * (1.0 - fu) * (1.0 - fv)
            + grid[v0, u1] * fu * (1.0 - fv)
            + grid[v1, u0] * (1.0 - fu) * fv
            + grid[v1, u1] * fu * fv
        )
        return np.asarray(np.where(inside, out, fill))


class PlanarThermalField:
    """A :class:`PlanarPatch` solved on a fixed tick and queried at arbitrary points.

    Thin by design: the tick schedule, the between-tick interpolation and the rule that a query
    never advances the solve all belong to :class:`~irsim.thermal.field.ThermalField` and are
    reused rather than restated (that rule exists because a renderer asks many times per tick, and
    an answer that depended on how often it was asked would not look like an error). What this
    class adds is the spatial half: a patch to hang the cells on, and a sample.
    """

    def __init__(
        self,
        patch: PlanarPatch,
        properties: FacetProperties,
        forcing_at: Callable[[float], FacetForcing],
        t0_s: float,
        initial_k: Any,
        tick_s: float = DEFAULT_TICK_S,
        *,
        keep_ticks: int | None = DEFAULT_KEEP_TICKS,
        on_tick: Callable[[float, NDArray[np.float64]], None] | None = None,
        conduction: Any = None,
        film_kg_m2: Any = None,
    ) -> None:
        if properties.n_facets != patch.n_cells:
            raise ValueError(
                f"properties describe {properties.n_facets} facets but the patch has "
                f"{patch.n_cells} cells"
            )
        state = np.asarray(initial_k, dtype=np.float64)
        if state.ndim == 0:
            state = np.full(patch.n_cells, float(state))
        self.patch = patch
        # A spatial field keeps a two-tick ring by default (PT.8, ADR 0093): a renderer asks for
        # the tick it just advanced to, never for last hour, and a 10 400-cell road held every
        # tick of a day at ~240 MB. Pass keep_ticks=None for the old behaviour, or on_tick to
        # record the history a time-lapse wants without holding it in the field.
        self.field = ThermalField(
            properties,
            forcing_at,
            t0_s,
            state,
            tick_s,
            keep_ticks=keep_ticks,
            on_tick=on_tick,
            conduction=conduction,
            film_kg_m2=film_kg_m2,
        )

    # -- delegation --------------------------------------------------------------------------

    @property
    def t0_s(self) -> float:
        return self.field.t0_s

    @property
    def tick_s(self) -> float:
        return self.field.tick_s

    @property
    def latest_t_s(self) -> float:
        return self.field.latest_t_s

    @property
    def n_ticks(self) -> int:
        return self.field.n_ticks

    @property
    def n_held(self) -> int:
        return self.field.n_held

    def advance_to(self, t_s: float) -> None:
        """Produce ticks up to ``t_s``. The **only** method that changes anything."""
        self.field.advance_to(t_s)

    def state_hash(self) -> str:
        return self.field.state_hash()

    @property
    def film_kg_m2(self) -> NDArray[np.float64] | None:
        """The water film per cell at the latest tick (PH.1), or ``None`` without one."""
        return self.field.film_kg_m2

    @property
    def evaporated_kg_m2(self) -> NDArray[np.float64]:
        """Water each cell's film has given up since t₀, kg m⁻² (PH.2)."""
        return self.field.evaporated_kg_m2

    # -- the query ---------------------------------------------------------------------------

    def temperature_at(self, t_s: float) -> NDArray[np.float32]:
        """``(n_cells,)`` float32 cell temperatures; reshape to ``patch.shape`` for an image."""
        return self.field.temperature_at(t_s)

    def temperature_image(self, t_s: float) -> NDArray[np.float32]:
        """The same values as ``(n_v, n_u)`` -- the surface seen face-on."""
        return np.asarray(self.temperature_at(t_s).reshape(self.patch.shape), dtype=np.float32)

    def sample_at(
        self, t_s: float, points: Any, *, fill: float = float("nan")
    ) -> NDArray[np.float32]:
        """Temperature at arbitrary points on the surface, float32 (CLAUDE.md #2)."""
        return np.asarray(
            self.patch.sample(self.temperature_at(t_s), points, fill=fill), dtype=np.float32
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"PlanarThermalField(frame={self.patch.frame!r}, "
            f"{self.patch.n_u}x{self.patch.n_v} cells, t={self.latest_t_s:.1f}s)"
        )
