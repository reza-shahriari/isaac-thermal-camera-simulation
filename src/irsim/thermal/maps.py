"""Temperature and parameter rasters draped onto a patch (PT.13).

DIRSIG has two escape hatches from solving a surface, and this project needs both for the same
reason it needs them: a skin whose temperature somebody else already knows.

* the **Map Temperature Solver** takes a single-band raster in degrees Celsius and projects it
  onto geometry, so the surface *is* the map and no energy balance runs on it;
* **MappedTherm** does the same for a material *parameter*, so one prim can carry a painted
  panel and a bare one without being two prims.

`PlanarPatch` is already a raster with a projection -- ``n_v`` by ``n_u`` cells with a known
origin and two axes -- so the projection is a resample and nothing more. That is deliberate: the
value of these hatches is that they are cheap, and a raster that had to be traced onto geometry
would not be.

**Where a map is the right answer.** A prescribed aerial skin whose temperature came from a wind
tunnel or a manufacturer; a hull solved in somebody else's CFD; and -- the one that matters for
this project's validation lane -- a **public thermal frame draped onto geometry**, so a scene can
be built from a real measurement rather than from this simulator's own solver. That last case is
also the reason the unit guard below exists: the published frames are in Celsius far more often
than in kelvin, and both are plain float rasters that load without complaint.

**Units are declared, never inferred, and the declaration is checked.** A 20 degC raster read as
kelvin is 20 K: not a cold surface, a physically impossible one, and the resulting frame is a
uniform floor rather than an error. The check is `LUT_T_MIN_K`, because a temperature the band
LUTs cannot evaluate is not a temperature this simulator can render -- a bound with a reason
rather than a round number. It is **asymmetric and says so**: kelvin mislabelled Celsius lands
300 K higher and inside the range, where nothing but a reader can catch it.

docs/physics-model.md §6.1, §13.1; roadmap PT.13; ADR 0087 (the field this drapes onto).
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import LUT_T_MAX_K, LUT_T_MIN_K
from irsim.thermal.facets import FacetProperties
from irsim.thermal.surface_field import PlanarPatch

__all__ = [
    "TemperatureUnits",
    "MappedParameter",
    "resample_to_patch",
    "temperature_map_to_cells",
    "parameter_map_to_cells",
    "apply_parameter_maps",
    "MapEntry",
    "PrescribedPatchField",
    "load_raster",
]

#: What a temperature raster may be labelled. There is no default: the whole point is that the
#: publisher's unit is a fact about the file and not something this code is entitled to assume.
TemperatureUnits = Literal["K", "degC"]

#: The `FacetProperties` fields a raster may drive, and the closed interval each must lie in.
#: Emissivity and solar absorptivity are fractions; a capacity is positive and unbounded above.
MappedParameter = Literal["emissivity", "solar_absorptivity", "heat_capacity_j_m2_k"]

_PARAMETER_RANGE: dict[str, tuple[float, float]] = {
    "emissivity": (0.0, 1.0),
    "solar_absorptivity": (0.0, 1.0),
    "heat_capacity_j_m2_k": (float(np.finfo(np.float64).tiny), float("inf")),
}


def load_raster(
    path: str | os.PathLike[str], *, data_dir: str | os.PathLike[str] | None = None
) -> NDArray[np.float64]:
    """A 2-D float raster from a ``.npy`` file, in float64 and refusing float16.

    Refusing half precision is CLAUDE.md non-negotiable #2 at the one boundary a *map* crosses:
    fp16 spaces values 0.25 K apart at 300 K, five times a 50 mK NETD, and a map is authored
    precisely when the caller has a better answer than the solver's.
    """
    from irsim.config.loader import resolve_data_dir

    p = pathlib.Path(path)
    if not p.is_absolute():
        p = resolve_data_dir(data_dir) / p
    arr = np.load(p)
    if arr.dtype == np.float16:
        raise TypeError(
            f"{p} is float16; a map is authored because somebody knows better than the solver, "
            "and half precision throws that away (CLAUDE.md #2). Store it as float32."
        )
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"{p} has dtype {arr.dtype}; a map must be a float raster")
    if arr.ndim != 2:
        raise ValueError(f"{p} has shape {arr.shape}; a map is a single-band 2-D raster")
    return np.asarray(arr, dtype=np.float64)


def resample_to_patch(values: Any, patch: PlanarPatch) -> NDArray[np.float64]:
    """``(rows, cols)`` raster -> ``(n_cells,)`` in the patch's own C order (v slow, u fast).

    **Row 0 is v = 0 and column 0 is u = 0**, so a raster shaped exactly ``(n_v, n_u)`` is the
    patch cell for cell and is returned with **no arithmetic at all** -- which is what makes the
    round trip exact rather than exact-to-a-tolerance. Any other shape is bilinear at the cells'
    own normalised centres, clamped at the border.

    The convention is stated rather than guessed because the alternative (row 0 at the far edge,
    as an image viewer draws it) is a vertical flip that looks like a plausible thermal picture
    either way -- the same class of error as ADR 0014's transposed camera rotation.
    """
    raster = np.asarray(values, dtype=np.float64)
    if raster.ndim != 2:
        raise ValueError(f"a map is a single-band 2-D raster, got shape {raster.shape}")
    if not np.isfinite(raster).all():
        raise ValueError("a map must be finite everywhere; a NaN cell has no temperature")
    if raster.shape == patch.shape:
        return np.asarray(raster.reshape(-1))

    rows, cols = raster.shape
    if rows < 1 or cols < 1:
        raise ValueError(f"an empty raster cannot be resampled, got shape {raster.shape}")
    # Cell centres in the raster's pixel coordinates: the patch's normalised centre scaled onto
    # the raster grid, with the half-pixel offset that puts pixel 0's centre at 0.0.
    v = ((np.arange(patch.n_v, dtype=np.float64) + 0.5) / patch.n_v) * rows - 0.5
    u = ((np.arange(patch.n_u, dtype=np.float64) + 0.5) / patch.n_u) * cols - 0.5
    v = np.clip(v, 0.0, rows - 1.0)
    u = np.clip(u, 0.0, cols - 1.0)
    v0, u0 = np.floor(v).astype(np.intp), np.floor(u).astype(np.intp)
    v1 = np.minimum(v0 + 1, rows - 1)
    u1 = np.minimum(u0 + 1, cols - 1)
    fv, fu = (v - v0)[:, None], (u - u0)[None, :]
    top = raster[np.ix_(v0, u0)] * (1.0 - fu) + raster[np.ix_(v0, u1)] * fu
    bottom = raster[np.ix_(v1, u0)] * (1.0 - fu) + raster[np.ix_(v1, u1)] * fu
    return np.asarray((top * (1.0 - fv) + bottom * fv).reshape(-1))


def temperature_map_to_cells(
    values: Any, patch: PlanarPatch, *, units: TemperatureUnits
) -> NDArray[np.float64]:
    """A temperature raster as ``(n_cells,)`` kelvin, with its declared unit checked.

    ``units`` has no default on purpose. The guard is the band LUTs' own domain: a cell outside
    `LUT_T_MIN_K`..`LUT_T_MAX_K` cannot be turned into radiance by anything downstream, so a map
    carrying one is refused here, where the file is named, rather than at a lookup that has
    forgotten which raster it came from.
    """
    if units == "K":
        kelvin = resample_to_patch(values, patch)
    elif units == "degC":
        kelvin = resample_to_patch(values, patch) + 273.15
    else:  # pragma: no cover - Literal is checked by mypy and by the config schema
        raise ValueError(f"unknown temperature units {units!r}; use 'K' or 'degC'")

    lo, hi = float(kelvin.min()), float(kelvin.max())
    if lo < LUT_T_MIN_K or hi > LUT_T_MAX_K:
        hint = ""
        if units == "K" and lo + 273.15 >= LUT_T_MIN_K and hi + 273.15 <= LUT_T_MAX_K:
            hint = (
                " Read as degC the same raster spans "
                f"{lo + 273.15:.2f}-{hi + 273.15:.2f} K, which is in range -- this looks like a "
                "Celsius raster declared as kelvin."
            )
        raise ValueError(
            f"a temperature map declared in {units} spans {lo:.2f}-{hi:.2f} K, outside the band "
            f"LUTs' {LUT_T_MIN_K:.0f}-{LUT_T_MAX_K:.0f} K domain, so no cell of it could be "
            f"turned into radiance.{hint}"
        )
    return kelvin


def parameter_map_to_cells(
    values: Any, patch: PlanarPatch, *, parameter: MappedParameter
) -> NDArray[np.float64]:
    """A parameter raster as ``(n_cells,)``, range-checked against what the quantity can be."""
    if parameter not in _PARAMETER_RANGE:
        raise ValueError(
            f"{parameter!r} is not a mappable parameter; use one of {sorted(_PARAMETER_RANGE)}"
        )
    cells = resample_to_patch(values, patch)
    lo, hi = _PARAMETER_RANGE[parameter]
    if float(cells.min()) < lo or float(cells.max()) > hi:
        raise ValueError(
            f"a {parameter} map spans {cells.min():.4g}-{cells.max():.4g}, outside [{lo:.4g}, "
            f"{hi:.4g}]. Kirchhoff closure is asserted on the material library, not recovered "
            "from a raster, so a map that leaves the interval is an authoring error."
        )
    return cells


def apply_parameter_maps(cells: FacetProperties, patch: PlanarPatch, maps: Any) -> FacetProperties:
    """A copy of ``cells`` with each named parameter replaced by its raster.

    Additive and ordered: a parameter named twice takes the last map, and a parameter named not
    at all keeps the library's value for every cell, so a surface with no maps is bit-identical
    to what it was before this existed.
    """
    if not maps:
        return cells
    fields = {
        "emissivity": np.array(cells.emissivity, dtype=np.float64, copy=True),
        "solar_absorptivity": np.array(cells.solar_absorptivity, dtype=np.float64, copy=True),
        "heat_capacity_j_m2_k": np.array(cells.heat_capacity_j_m2_k, dtype=np.float64, copy=True),
    }
    for entry in maps:
        fields[entry.parameter] = parameter_map_to_cells(
            entry.values, patch, parameter=entry.parameter
        )
    return FacetProperties(**fields)


@dataclass(frozen=True)
class MapEntry:
    """What :func:`apply_parameter_maps` needs of a config row, without importing the schema."""

    parameter: MappedParameter
    values: NDArray[np.float64]


class PrescribedPatchField:
    """A patch whose temperature is a map and not a solve (DIRSIG's Map Temperature Solver).

    It carries the same surface as `PlanarThermalField` and `PatchView` so the point bridge, the
    render path and every test see one kind of thing. Nothing advances: `advance_to` is accepted
    and ignored, because a caller that has to know which of its surfaces are solved and which are
    prescribed is a caller that will eventually get it wrong.

    Static in time by design. A map is a measurement of one moment; interpolating between two of
    them would invent a thermal history that the file does not contain, and the honest way to get
    a time-varying prescribed skin is two scenes.
    """

    def __init__(self, patch: PlanarPatch, temperatures_k: Any, t0_s: float = 0.0) -> None:
        cells = np.asarray(temperatures_k, dtype=np.float64).reshape(-1)
        if cells.size != patch.n_cells:
            raise ValueError(f"the map has {cells.size} cells and the patch has {patch.n_cells}")
        self.patch = patch
        self.t0_s = float(t0_s)
        self.tick_s = float("inf")
        self._cells = cells

    @property
    def latest_t_s(self) -> float:
        """A prescribed field is valid at every time, so it is never behind a query."""
        return float("inf")

    def advance_to(self, t_s: float) -> None:
        del t_s  # nothing to advance: see the class docstring

    def temperature_at(self, t_s: float) -> NDArray[np.float32]:
        del t_s
        return np.asarray(self._cells, dtype=np.float32)

    def temperature_image(self, t_s: float) -> NDArray[np.float32]:
        return np.asarray(self.temperature_at(t_s).reshape(self.patch.shape), dtype=np.float32)

    def sample_at(
        self, t_s: float, points: Any, *, fill: float = float("nan")
    ) -> NDArray[np.float32]:
        return np.asarray(
            self.patch.sample(self.temperature_at(t_s), points, fill=fill), dtype=np.float32
        )
