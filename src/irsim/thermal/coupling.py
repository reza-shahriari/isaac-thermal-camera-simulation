"""Contactors and radiation between fields: two surfaces that touch, or face each other (TC.3).

docs/physics-model.md §6.1, §6.4; ADR 0087 (the field), ADR 0088 (the view factors), ADR 0094
(the implicit step this rides on), ADR 0099 (this module).

A bonnet skin is bolted, clipped and gasketed to the body under it; a road patch faces the
underbody of the car above it. TC.1 gave one field a conduction term and TC.2 gave lumped parts
a network; this module joins **fields to each other**, two ways:

**A contactor** is a conductance between the cells of two patches that overlap, proportional to
the overlap area: ``K_ij = h_c · A_ij`` with ``A_ij`` the area cell *i* of one patch shares with
cell *j* of the other, computed by clipping the second cell's rectangle against the first's in
the first patch's own plane. The grids need not align, need not share a cell size and need not
share an orientation; the total conductance is ``h_c · A_overlap`` to clipping precision, and it
does not change when one grid is refined -- which a conductance authored *per node* cannot do
(refine the grid and a per-node total scales with the cell count). ``h_c`` is a contact
conductance from `configs/thermal/joints.yaml` or an inline value in W m⁻² K⁻¹.

**Coupled fields are one solver.** Conduction at a bolted joint is stiff -- a 5 mm cell at
10⁴ W m⁻² K⁻¹ has a time constant under a second against a 60 s tick -- so an exchange stepped
explicitly between two separate fields would blow up, and a lagged one would ring. Instead the
patches' cells are concatenated into **one** `ThermalField` with a block conduction operator
(each member's own lateral operator on the diagonal, the contactors off it) and stepped by
ADR 0094's IMEX scheme; each member is then a :class:`PatchView` over its slice, which is all
the point bridge needs (``patch``, ``advance_to``, ``sample_at``). Nothing in the integrator
changes, and a coupled system with no contactor is the separate fields, bit for bit.

Since TC.16 a member's patch may be a `TriangleMeshPatch` as well as a `PlanarPatch`: the
only thing the coupled solve ever asked of a patch is ``n_cells`` and ``cell_area_m2``, and a
mesh answers with an array where a grid answers with one number. Two more ways to join members
come with it, because rectangle clipping cannot see a tube: an :class:`ExplicitContactor` is a
conductance matrix built elsewhere -- by :func:`proximity_contactor`, which pairs the cells of
two parts that stand within a gap of each other and spreads an authored contact area over them
-- and a :class:`LumpedLink` may carry per-cell conductances, from :func:`footprint_conductances`,
so a hidden part (AI.11) is bolted to the cells under its footprint and not to the whole skin.

**Radiation between a body and a field** reuses ADR 0088's parallel-rectangle view factors
(`spatial_sources`), read in both directions. From the cells' side, :class:`RadiationExchange`
is the net flux `occluded_longwave_flux` already gives a bonnet under a bay or asphalt under a
sill; from the body's side, reciprocity ``A_i F_i→r = A_r F_r→i`` makes the power the body
loses to the patch the sum of what the cells receive, so an engine → bonnet exchange and an
underbody → road exchange are one mechanism, and a network node (TC.5) can be charged for
what its face radiates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.conduction import ConductionOperator
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.field import DEFAULT_TICK_S, ThermalField
from irsim.thermal.spatial_sources import (
    RadiantRectangle,
    occluded_longwave_flux,
    patch_view_factors,
    view_factor_to_parallel_rectangle,
)
from irsim.thermal.surface_field import DEFAULT_KEEP_TICKS, PlanarPatch

__all__ = [
    "ExplicitContactor",
    "LumpedLink",
    "LumpedMember",
    "Contactor",
    "CoupledFields",
    "FieldMember",
    "PatchView",
    "footprint_conductances",
    "proximity_contactor",
    "RadiationExchange",
    "cell_overlap_areas",
    "contactor_conductances",
]


# ---------------------------------------------------------------------------------------------
# overlap geometry
# ---------------------------------------------------------------------------------------------


def _clip_polygon_to_box(
    poly: list[tuple[float, float]], u0: float, u1: float, v0: float, v1: float
) -> list[tuple[float, float]]:
    """Sutherland–Hodgman: a convex polygon clipped to an axis-aligned box in (u, v)."""

    def clip(
        points: list[tuple[float, float]], inside: Any, intersect: Any
    ) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        n = len(points)
        for k in range(n):
            current, previous = points[k], points[k - 1]
            if inside(current):
                if not inside(previous):
                    out.append(intersect(previous, current))
                out.append(current)
            elif inside(previous):
                out.append(intersect(previous, current))
        return out

    def cross_u(level: float) -> Any:
        def f(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
            t = (level - p[0]) / (q[0] - p[0])
            return (level, p[1] + t * (q[1] - p[1]))

        return f

    def cross_v(level: float) -> Any:
        def f(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
            t = (level - p[1]) / (q[1] - p[1])
            return (p[0] + t * (q[0] - p[0]), level)

        return f

    poly = clip(poly, lambda p: p[0] >= u0, cross_u(u0))
    if poly:
        poly = clip(poly, lambda p: p[0] <= u1, cross_u(u1))
    if poly:
        poly = clip(poly, lambda p: p[1] >= v0, cross_v(v0))
    if poly:
        poly = clip(poly, lambda p: p[1] <= v1, cross_v(v1))
    return poly


def _area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for k in range(len(poly)):
        (x0, y0), (x1, y1) = poly[k - 1], poly[k]
        s += x0 * y1 - x1 * y0
    return 0.5 * abs(s)


def cell_overlap_areas(
    patch_a: PlanarPatch, patch_b: PlanarPatch, *, gap_tolerance_m: float = 0.05
) -> Any:
    """``(n_a, n_b)`` sparse matrix of the area (m²) cell *i* of ``patch_a`` shares with cell *j*
    of ``patch_b``, both projected onto ``patch_a``'s plane.

    The patches must be parallel (their normals aligned or opposed) and within
    ``gap_tolerance_m`` of each other along the normal: a bonnet skin over its frame is a few
    millimetres, a gasket a centimetre; a bonnet 0.9 m above a road is not in contact and the
    call refuses rather than clipping two rectangles that happen to project onto each other.
    """
    if patch_a.frame != patch_b.frame:
        raise ValueError(
            f"the patches are in different frames ({patch_a.frame!r}, {patch_b.frame!r}); a "
            "contactor needs both in one frame"
        )
    if abs(abs(float(np.dot(patch_a.normal, patch_b.normal))) - 1.0) > 1e-9:
        raise ValueError("a contactor joins parallel patches; these are not parallel")
    corners_b = _cell_corners(patch_b)  # (n_b, 4, 3)
    uvw = patch_a.local_coords(corners_b.reshape(-1, 3)).reshape(patch_b.n_cells, 4, 3)
    gap = float(np.max(np.abs(uvw[..., 2])))
    if gap > gap_tolerance_m:
        raise ValueError(
            f"the patches are {gap:.4f} m apart along the normal, more than the contact "
            f"tolerance {gap_tolerance_m} m: not a joint"
        )
    du, dv = patch_a.du_m, patch_a.dv_m
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for j in range(patch_b.n_cells):
        poly = [(float(u), float(v)) for u, v in uvw[j, :, :2]]
        us = [p[0] for p in poly]
        vs = [p[1] for p in poly]
        iu0 = max(0, int(np.floor(min(us) / du)))
        iu1 = min(patch_a.n_u - 1, int(np.floor(max(us) / du)))
        iv0 = max(0, int(np.floor(min(vs) / dv)))
        iv1 = min(patch_a.n_v - 1, int(np.floor(max(vs) / dv)))
        for iv in range(iv0, iv1 + 1):
            for iu in range(iu0, iu1 + 1):
                area = _area(
                    _clip_polygon_to_box(poly, iu * du, (iu + 1) * du, iv * dv, (iv + 1) * dv)
                )
                if area > 0.0:
                    rows.append(iv * patch_a.n_u + iu)
                    cols.append(j)
                    vals.append(area)
    return sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(patch_a.n_cells, patch_b.n_cells),
    )


def _cell_corners(patch: PlanarPatch) -> NDArray[np.float64]:
    """``(n_cells, 4, 3)`` world-frame corners of every cell, counter-clockwise in (u, v)."""
    iu = np.arange(patch.n_u, dtype=np.float64)
    iv = np.arange(patch.n_v, dtype=np.float64)
    gu, gv = np.meshgrid(iu, iv)  # (n_v, n_u)
    u0 = (gu * patch.du_m).reshape(-1)
    v0 = (gv * patch.dv_m).reshape(-1)
    offsets = np.array([[0.0, 0.0], [patch.du_m, 0.0], [patch.du_m, patch.dv_m], [0.0, patch.dv_m]])
    uv = np.stack([u0, v0], axis=-1)[:, None, :] + offsets[None, :, :]  # (n, 4, 2)
    return np.asarray(
        patch.origin_m
        + uv[..., 0:1] * patch.u_axis[None, None, :]
        + uv[..., 1:2] * patch.v_axis[None, None, :]
    )


def contactor_conductances(
    patch_a: PlanarPatch,
    patch_b: PlanarPatch,
    h_c_w_m2_k: float,
    *,
    gap_tolerance_m: float = 0.05,
) -> Any:
    """``(n_a, n_b)`` link conductances in W/K: ``h_c`` times the overlap of each cell pair."""
    if h_c_w_m2_k < 0.0:
        raise ValueError("a contact conductance cannot be negative")
    return cell_overlap_areas(patch_a, patch_b, gap_tolerance_m=gap_tolerance_m) * float(h_c_w_m2_k)


# ---------------------------------------------------------------------------------------------
# coupled fields
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldMember:
    """One patch in a coupled solve: its cells, their forcing, and its own lateral operator."""

    name: str
    patch: Any  # a PlanarPatch, or since TC.16 a TriangleMeshPatch
    properties: FacetProperties
    forcing_at: Callable[[float], FacetForcing]
    initial_k: Any
    conduction: Any = None  # an intra-patch ConductionOperator (PT.11), or None

    def __post_init__(self) -> None:
        if self.properties.n_facets != self.patch.n_cells:
            raise ValueError(
                f"member {self.name!r}: properties describe {self.properties.n_facets} facets, "
                f"the patch has {self.patch.n_cells} cells"
            )
        if self.conduction is not None and self.conduction.n_facets != self.patch.n_cells:
            raise ValueError(
                f"member {self.name!r}: its conduction operator does not fit its patch"
            )


@dataclass(frozen=True)
class Contactor:
    """A contact between two members' patches at ``h_c`` W m⁻² K⁻¹ over their overlap."""

    a: str
    b: str
    h_c_w_m2_k: float
    gap_tolerance_m: float = 0.05

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError("a contactor joins two different members")
        if self.h_c_w_m2_k < 0.0:
            raise ValueError("a contact conductance cannot be negative")


@dataclass(frozen=True)
class ExplicitContactor:
    """A contact whose conductances are already a matrix: ``(n_a, n_b)`` W/K, cell to cell.

    For the geometries rectangle clipping cannot express -- a motor can on a mesh arm, a pack
    on a shell -- built by :func:`proximity_contactor` (TC.16) or by hand in a test.
    """

    a: str
    b: str
    conductance_w_k: Any

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError("a contactor joins two different members")
        k = sp.csr_matrix(self.conductance_w_k, dtype=np.float64)
        if k.shape[0] == 0 or k.shape[1] == 0 or (k.data < 0.0).any():
            raise ValueError("an explicit contactor is a non-negative (n_a, n_b) matrix")
        object.__setattr__(self, "conductance_w_k", k)


def _areas_of(patch: Any, n: int) -> NDArray[np.float64]:
    """``(n,)`` cell areas: a grid's one number broadcast, a mesh's own array (TC.16)."""
    return np.array(np.broadcast_to(np.asarray(patch.cell_area_m2, dtype=np.float64), (n,)))


def proximity_contactor(
    patch_a: Any,
    patch_b: Any,
    h_c_w_m2_k: float,
    area_m2: float,
    *,
    gap_m: float,
) -> Any:
    """``(n_a, n_b)`` W/K between the cells of two parts that stand within ``gap_m`` (TC.16).

    Every pair of cells whose centres lie within ``gap_m`` shares the contact, weighted by the
    smaller of the two cell areas, and the weights are scaled so the whole matrix sums to
    ``h_c · area_m2`` -- the authored footprint (AI.11's `ContactSpec`) is what conducts, not
    however many cells happened to fall inside the gap. Refuses when no cell pair is that close:
    two parts that do not touch cannot be bolted together by a number.
    """
    if h_c_w_m2_k < 0.0 or area_m2 <= 0.0 or gap_m <= 0.0:
        raise ValueError("proximity_contactor needs h_c >= 0, area_m2 > 0 and gap_m > 0")
    ca = np.asarray(patch_a.cell_centres(), dtype=np.float64)
    cb = np.asarray(patch_b.cell_centres(), dtype=np.float64)
    aa, ab = _areas_of(patch_a, ca.shape[0]), _areas_of(patch_b, cb.shape[0])
    d2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(axis=-1)
    i, j = np.nonzero(d2 <= gap_m * gap_m)
    if i.size == 0:
        raise ValueError(
            f"no cell of one part lies within {gap_m} m of a cell of the other: the parts do "
            "not touch, so there is nothing for a contact to conduct through"
        )
    w = np.minimum(aa[i], ab[j])
    w = w / w.sum() * float(area_m2) * float(h_c_w_m2_k)
    return sp.csr_matrix((w, (i, j)), shape=(ca.shape[0], cb.shape[0]))


def footprint_conductances(
    patch: Any, centre_m: Any, area_m2: float, h_c_w_m2_k: float
) -> NDArray[np.float64]:
    """``(n_cells,)`` W/K from a lumped part to the cells under its footprint (TC.16).

    Cells are taken nearest-first from ``centre_m`` until their area reaches ``area_m2`` (the
    last one pro rata), each conducting ``h_c × its share``; the sum is ``h_c · area_m2``
    exactly. This is how a hidden part (AI.11) meets the skin it is bolted to.
    """
    if area_m2 <= 0.0 or h_c_w_m2_k < 0.0:
        raise ValueError("footprint_conductances needs area_m2 > 0 and h_c >= 0")
    centres = np.asarray(patch.cell_centres(), dtype=np.float64)
    areas = _areas_of(patch, centres.shape[0])
    order = np.argsort(((centres - np.asarray(centre_m, dtype=np.float64)) ** 2).sum(axis=1))
    out = np.zeros(centres.shape[0])
    remaining = float(area_m2)
    for idx in order:
        take = min(remaining, float(areas[idx]))
        out[idx] = take * float(h_c_w_m2_k)
        remaining -= take
        if remaining <= 0.0:
            break
    if remaining > 1e-12 * float(area_m2):
        raise ValueError(
            f"the footprint ({area_m2} m2) is larger than the whole part ({areas.sum():.6g} m2)"
        )
    return out


@dataclass(frozen=True)
class LumpedMember:
    """A node with no cells -- a cabin's air, any fluid volume -- solved with the fields (PT.15).

    It is one cell of unit area: ``capacity_j_k`` is its J/K, its forcing's ``h_w_m2_k`` is a
    conductance in W/K to ``t_air_k`` (infiltration to ambient), ``q_internal_w_m2`` is watts
    (solar through the glazing), and ε = α = 0 so it neither radiates nor sees the sun. Joined
    to a field's cells by :class:`LumpedLink`s, which is how a panel gets a back boundary that
    is neither adiabatic nor ambient (ADR 0038, ADR 0106).
    """

    name: str
    capacity_j_k: float
    forcing_at: Callable[[float], FacetForcing]
    initial_k: float

    def __post_init__(self) -> None:
        if self.capacity_j_k <= 0.0:
            raise ValueError(f"lumped member {self.name!r}: capacity must be positive (J/K)")
        if self.initial_k <= 0.0:
            raise ValueError(f"lumped member {self.name!r}: initial_k must be positive")

    def as_member(self) -> FieldMember:
        patch = PlanarPatch(
            origin_m=np.zeros(3),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 1.0, 0.0]),
            n_u=1,
            n_v=1,
            du_m=1.0,
            dv_m=1.0,
            thickness_m=1.0,
            frame=f"lumped:{self.name}",
        )
        props = FacetProperties(
            heat_capacity_j_m2_k=np.array([self.capacity_j_k]),
            emissivity=np.array([0.0]),
            solar_absorptivity=np.array([0.0]),
        )
        return FieldMember(self.name, patch, props, self.forcing_at, np.array([self.initial_k]))


@dataclass(frozen=True)
class LumpedLink:
    """``h · A_cell`` from every cell of member ``field`` to the lumped ``node``, W/K per cell.

    ``h_w_m2_k`` is ``1/R`` for a panel's inner resistance (conduction plus the interior film,
    §6.6 / ADR 0038): the flux a cell loses inward is ``(T_cell − T_node)/R`` and the node
    gains ``Σ A_cell (T_cell − T_node)/R`` -- `CabinNode`'s two terms, on one operator.
    """

    field: str
    node: str
    h_w_m2_k: float
    #: TC.16: per-cell conductances in W/K that replace ``h · A_cell`` -- a footprint from
    #: :func:`footprint_conductances`, zero on the cells the part does not touch.
    per_cell_w_k: Any = None

    def __post_init__(self) -> None:
        if self.field == self.node:
            raise ValueError("a lumped link joins a field to a different node")
        if self.h_w_m2_k < 0.0:
            raise ValueError("a lumped link's conductance cannot be negative")
        if self.per_cell_w_k is not None:
            arr = np.asarray(self.per_cell_w_k, dtype=np.float64)
            if arr.ndim != 1 or (arr < 0.0).any():
                raise ValueError("per_cell_w_k is a non-negative (n_cells,) array")
            object.__setattr__(self, "per_cell_w_k", arr)


class PatchView:
    """One member's slice of a :class:`CoupledFields`: what the point bridge and a test see."""

    def __init__(self, coupled: CoupledFields, patch: Any, start: int, stop: int) -> None:
        self._coupled = coupled
        self.patch = patch
        self._slice = slice(start, stop)

    @property
    def field(self) -> ThermalField:
        """The shared solver; queries through it return **every** member's cells."""
        return self._coupled.field

    @property
    def cells(self) -> slice:
        """This member's cells in the shared solver's state."""
        return self._slice

    @property
    def internal_exchange(self) -> bool:
        """Whether the solve already exchanges radiation between its own members (TC.11)."""
        return self._coupled.exchange is not None

    @property
    def t0_s(self) -> float:
        return self._coupled.field.t0_s

    @property
    def tick_s(self) -> float:
        return self._coupled.field.tick_s

    @property
    def latest_t_s(self) -> float:
        return self._coupled.field.latest_t_s

    def advance_to(self, t_s: float) -> None:
        """Advances the whole coupled solve: every member moves together, by construction."""
        self._coupled.advance_to(t_s)

    def temperature_at(self, t_s: float) -> NDArray[np.float32]:
        return np.asarray(self._coupled.field.temperature_at(t_s)[self._slice], dtype=np.float32)

    def temperature_image(self, t_s: float) -> NDArray[np.float32]:
        if not hasattr(self.patch, "shape"):
            raise AttributeError("a mesh member has no image; read temperature_at")
        return np.asarray(self.temperature_at(t_s).reshape(self.patch.shape), dtype=np.float32)

    def sample_at(
        self, t_s: float, points: Any, *, fill: float = float("nan")
    ) -> NDArray[np.float32]:
        return np.asarray(
            self.patch.sample(self.temperature_at(t_s), points, fill=fill), dtype=np.float32
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"PatchView({self.patch.n_u}x{self.patch.n_v} cells of a coupled solve)"


class CoupledFields:
    """Several patches stepped as one `ThermalField`, joined by contactors (ADR 0099)."""

    def __init__(
        self,
        members: Sequence[FieldMember],
        contactors: Sequence[Contactor | ExplicitContactor] = (),
        *,
        t0_s: float,
        tick_s: float = DEFAULT_TICK_S,
        keep_ticks: int | None = DEFAULT_KEEP_TICKS,
        on_tick: Callable[[float, NDArray[np.float64]], None] | None = None,
        lumped: Sequence[LumpedMember] = (),
        lumped_links: Sequence[LumpedLink] = (),
        exchange: Any = None,
        exchange_members: Sequence[str] = (),
    ) -> None:
        if not members and not lumped:
            raise ValueError("a coupled solve needs at least one member")
        # TC.11: radiation between members by TC.9's factors, added to q_internal from the
        # start-of-tick state -- explicit and one tick lagged, as ADR 0157 argues is stable for
        # a radiative coupling, inside the implicit conduction step.
        self.exchange = exchange
        self.exchange_members: tuple[str, ...] = tuple(exchange_members)
        if (exchange is None) != (not self.exchange_members):
            raise ValueError("exchange and exchange_members come together")
        members = tuple(members) + tuple(m.as_member() for m in lumped)
        self.lumped_names: tuple[str, ...] = tuple(m.name for m in lumped)
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError(f"member names must be unique: {names}")
        self.members = tuple(members)
        self._index = {m.name: i for i, m in enumerate(members)}
        sizes = [m.patch.n_cells for m in members]
        starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        self._slices = {m.name: (int(starts[i]), int(starts[i + 1])) for i, m in enumerate(members)}
        properties = FacetProperties(
            heat_capacity_j_m2_k=np.concatenate(
                [m.properties.heat_capacity_j_m2_k for m in members]
            ),
            emissivity=np.concatenate([m.properties.emissivity for m in members]),
            solar_absorptivity=np.concatenate([m.properties.solar_absorptivity for m in members]),
        )
        initial = np.concatenate(
            [
                np.broadcast_to(np.asarray(m.initial_k, dtype=np.float64), (m.patch.n_cells,))
                for m in members
            ]
        )
        areas = np.concatenate([_areas_of(m.patch, m.patch.n_cells) for m in members])
        n = int(starts[-1])
        blocks: dict[tuple[int, int], Any] = {}
        for i, m in enumerate(members):
            if m.conduction is not None:
                blocks[(i, i)] = m.conduction.conductance_w_k
        self.contactor_conductances: dict[tuple[str, str], Any] = {}
        for c in contactors:
            for end in (c.a, c.b):
                if end not in self._index:
                    raise ValueError(f"contactor names unknown member {end!r}; known: {names}")
            i, j = self._index[c.a], self._index[c.b]
            if isinstance(c, ExplicitContactor):
                k_ab = c.conductance_w_k
                if k_ab.shape != (sizes[i], sizes[j]):
                    raise ValueError(
                        f"contactor {c.a!r}-{c.b!r}: matrix is {k_ab.shape}, the members have "
                        f"{sizes[i]} and {sizes[j]} cells"
                    )
            else:
                for idx in (i, j):
                    if not isinstance(members[idx].patch, PlanarPatch):
                        raise ValueError(
                            f"contactor {c.a!r}-{c.b!r}: overlap clipping needs planar patches; "
                            f"{members[idx].name!r} is a mesh -- use an ExplicitContactor "
                            "(proximity_contactor)"
                        )
                k_ab = contactor_conductances(
                    members[i].patch,
                    members[j].patch,
                    c.h_c_w_m2_k,
                    gap_tolerance_m=c.gap_tolerance_m,
                )
            self.contactor_conductances[(c.a, c.b)] = k_ab
            blocks[(i, j)] = k_ab if (i, j) not in blocks else blocks[(i, j)] + k_ab
            blocks[(j, i)] = k_ab.T if (j, i) not in blocks else blocks[(j, i)] + k_ab.T
        for link in lumped_links:
            if link.field not in self._index or link.node not in self.lumped_names:
                raise ValueError(
                    f"lumped link {link.field!r} -> {link.node!r}: the field must be a member and "
                    f"the node a lumped member; known {names}, lumped {list(self.lumped_names)}"
                )
            i, j = self._index[link.field], self._index[link.node]
            if link.per_cell_w_k is not None:
                if link.per_cell_w_k.shape != (sizes[i],):
                    raise ValueError(
                        f"lumped link {link.field!r} -> {link.node!r}: per_cell_w_k has "
                        f"{link.per_cell_w_k.shape[0]} entries, the field {sizes[i]} cells"
                    )
                column = link.per_cell_w_k.reshape(-1, 1)
            else:
                column = (link.h_w_m2_k * _areas_of(members[i].patch, sizes[i])).reshape(-1, 1)
            k_ab = sp.csr_matrix(column)
            self.contactor_conductances[(link.field, link.node)] = k_ab
            blocks[(i, j)] = k_ab if (i, j) not in blocks else blocks[(i, j)] + k_ab
            blocks[(j, i)] = k_ab.T if (j, i) not in blocks else blocks[(j, i)] + k_ab.T
        conduction = None
        if blocks:
            grid = [
                [
                    blocks.get((i, j), sp.csr_matrix((sizes[i], sizes[j])))
                    for j in range(len(members))
                ]
                for i in range(len(members))
            ]
            conduction = ConductionOperator(sp.bmat(grid, format="csr"), areas)
        self._n = n
        self.field = ThermalField(
            properties,
            self._forcing,
            t0_s,
            initial,
            tick_s,
            keep_ticks=keep_ticks,
            on_tick=on_tick,
            conduction=conduction,
        )
        self.fields: Mapping[str, PatchView] = {
            m.name: PatchView(self, m.patch, *self._slices[m.name]) for m in members
        }
        if self.exchange is not None:
            if tuple(self.exchange.names) != self.exchange_members:
                raise ValueError(
                    f"the exchange's bodies {self.exchange.names} must be exchange_members "
                    f"{self.exchange_members}, in order"
                )
            for k, name in enumerate(self.exchange_members):
                if name not in self._index:
                    raise ValueError(f"exchange member {name!r} is not a member")
                lo, hi = self._slices[name]
                if hi - lo != self.exchange.bodies[k].n_cells:
                    raise ValueError(
                        f"exchange body {name!r} has {self.exchange.bodies[k].n_cells} cells, "
                        f"the member {hi - lo}"
                    )

    def _forcing(self, t_s: float) -> FacetForcing:
        """Every member's forcing at ``t_s``, concatenated cell for cell."""
        forcings = [m.forcing_at(t_s) for m in self.members]
        parts = [f.arrays(m.patch.n_cells) for f, m in zip(forcings, self.members, strict=True)]
        t_air, h, q_solar, q_lw, q_int = (np.concatenate(col) for col in zip(*parts, strict=True))
        leaves = np.concatenate(
            [
                f.emission_factors(m.patch.n_cells)
                for f, m in zip(forcings, self.members, strict=True)
            ]
        )
        if self.exchange is not None:
            q_int = self._with_exchange(q_int, q_lw)
        return FacetForcing(
            t_air_k=t_air,
            h_w_m2_k=h,
            q_solar_w_m2=q_solar,
            q_longwave_down_w_m2=q_lw,
            q_internal_w_m2=q_int,
            emission_factor=leaves,
        )

    def _with_exchange(
        self, q_int: NDArray[np.float64], q_lw: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """``q_internal`` plus what the exchange members radiate onto each other (TC.11)."""
        state = self.field.latest_state_k
        slices = [self._slices[name] for name in self.exchange_members]
        temps = np.concatenate([state[lo:hi] for lo, hi in slices])
        lw = np.concatenate([q_lw[lo:hi] for lo, hi in slices])
        q = self.exchange.flux_w_m2(temps, lw)
        out = np.array(q_int, dtype=np.float64)
        at = 0
        for lo, hi in slices:
            out[lo:hi] += q[at : at + hi - lo]
            at += hi - lo
        return out

    @property
    def n_cells(self) -> int:
        return self._n

    @property
    def cell_areas_m2(self) -> NDArray[np.float64]:
        return np.concatenate([_areas_of(m.patch, m.patch.n_cells) for m in self.members])

    def advance_to(self, t_s: float) -> None:
        self.field.advance_to(t_s)

    def temperature_at(self, t_s: float) -> NDArray[np.float32]:
        """All members' cells, in member order."""
        return self.field.temperature_at(t_s)

    def node_temperature_k(self, name: str) -> float:
        """A lumped member's temperature at the latest tick, in float64 from the solver."""
        if name not in self.lumped_names:
            raise KeyError(f"{name!r} is not a lumped member; lumped: {list(self.lumped_names)}")
        start, _stop = self._slices[name]
        return float(self.field.latest_state_k[start])

    def node_temperature_at(self, name: str, t_s: float) -> float:
        """A lumped member's temperature at ``t_s``, from the kept ticks (TC.13): what a test
        or a legend asks after the solve has moved on."""
        if name not in self.lumped_names:
            raise KeyError(f"{name!r} is not a lumped member; lumped: {list(self.lumped_names)}")
        start, _stop = self._slices[name]
        return float(self.field.temperature_at(t_s)[start])

    def stored_energy_j(self) -> float:
        """``Σ C_i A_i T_i`` at the latest tick, in float64 from the solver's own state."""
        temps = self.field.latest_state_k
        capacity = self.field.properties.heat_capacity_j_m2_k
        return float(np.sum(capacity * self.cell_areas_m2 * temps))


# ---------------------------------------------------------------------------------------------
# radiation between a body and a field
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RadiationExchange:
    """A radiating rectangle facing a patch: the cells' net flux, and the body's loss.

    ``surface_emissivity`` is the cells' (per cell or scalar); the rectangle's own is on it.
    The cells' term is `occluded_longwave_flux` (ADR 0088: the body also blocks the sky);
    the body's loss to the patch is ``ε_r σ T_r⁴ Σ_i A_i F_i``, which by reciprocity is
    ``A_r F_r→patch ε_r σ T_r⁴`` -- the same number seen from the other side.
    """

    rect: RadiantRectangle
    patch: PlanarPatch
    surface_emissivity: Any

    @property
    def view_factors(self) -> NDArray[np.float64]:
        """``(n_cells,)`` F from each cell to the rectangle (Howell C-11)."""
        return patch_view_factors(self.patch, self.rect)

    @property
    def rect_to_patch_view_factor(self) -> float:
        """``F_r→patch = Σ_i A_i F_i→r / A_r`` by reciprocity; at most 1."""
        return float(np.sum(self.view_factors) * self.patch.cell_area_m2 / self.rect.area_m2)

    def cell_flux_w_m2(
        self, source_temperature_k: float, *, longwave_down_w_m2: Any = 0.0, sky_view: Any = 1.0
    ) -> NDArray[np.float64]:
        """The net ``q_internal`` per cell: the body's radiation less the sky it blocks."""
        return occluded_longwave_flux(
            self.view_factors,
            source_temperature_k,
            self.surface_emissivity,
            source_emissivity=self.rect.emissivity,
            longwave_down_w_m2=longwave_down_w_m2,
            sky_view=sky_view,
        )

    def intercepted_power_w(self, source_temperature_k: float) -> float:
        """Power the rectangle radiates *onto* the patch, W: ``ε_r σ T_r⁴ Σ_i A_i F_i``."""
        if source_temperature_k <= 0.0:
            raise ValueError("source temperature must be positive (kelvin)")
        emitted = self.rect.emissivity * SIGMA_SB * source_temperature_k**4
        return float(emitted * np.sum(self.view_factors) * self.patch.cell_area_m2)

    def arriving_power_w(self, source_temperature_k: float) -> float:
        """The same power summed from the cells' side, W: ``Σ_i A_i F_i ε_r σ T_r⁴``."""
        f = self.view_factors
        emitted = self.rect.emissivity * SIGMA_SB * source_temperature_k**4
        return float(np.sum(f * emitted) * self.patch.cell_area_m2)

    def reverse_view_factor(self, n_u: int = 40, n_v: int = 40) -> float:
        """``F_r→patch`` by an independent quadrature from the rectangle's side.

        The rectangle is tiled ``n_u × n_v`` and each tile's view factor to the *patch's* whole
        rectangle is evaluated by the same closed form, so agreement with
        :attr:`rect_to_patch_view_factor` is a check of reciprocity between two discretisations,
        not of one formula against itself.
        """
        span_u, span_v = self.patch.extent_m
        target = RadiantRectangle(
            centre_m=self.patch.origin_m
            + 0.5 * span_u * self.patch.u_axis
            + 0.5 * span_v * self.patch.v_axis,
            u_axis=self.patch.u_axis,
            v_axis=self.patch.v_axis,
            half_u_m=0.5 * span_u,
            half_v_m=0.5 * span_v,
        )
        us = (np.arange(n_u) + 0.5) / n_u * 2.0 * self.rect.half_u_m - self.rect.half_u_m
        vs = (np.arange(n_v) + 0.5) / n_v * 2.0 * self.rect.half_v_m - self.rect.half_v_m
        gu, gv = np.meshgrid(us, vs)
        points = (
            self.rect.centre_m
            + gu.reshape(-1, 1) * self.rect.u_axis
            + gv.reshape(-1, 1) * self.rect.v_axis
        )
        return float(np.mean(view_factor_to_parallel_rectangle(points, target)))
