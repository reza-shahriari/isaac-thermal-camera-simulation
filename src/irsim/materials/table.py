"""Scalar-per-band material table: material id → in-band ε₀ (ρ, τ, angular, thermal columns).

The kernel-facing table (docs/physics-model.md §13.5 ``matEps``): contiguous **float32** arrays
indexed by material id for one band, built from the M7.2 library by
:meth:`MaterialTable.from_library` (Level C: ε₀ plus the Level-B (a, p) the file authored, or the
(0, 4) placeholders) or authored directly for tests. Id **0 is the UNMAPPED sentinel** (an asset
the material resolver could not map); a kernel that meets it must fail, not silently render a
default. Ids are assigned in sorted-name order so they are stable across loads; the ``.npz`` +
sidecar carries the library hash and a stale table is refused (ADR 0040, ADR 0012 pattern).
float16 anywhere in the table is refused (CLAUDE.md #2).

docs/physics-model.md §4.1, §12.3, §13.3, §13.5
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from irsim.config.sensor import SensorConfig, SensorSpec
    from irsim.materials.library import MaterialLibrary
    from irsim.radiometry.band_average import WeightingForm
    from irsim.radiometry.spectral_response import SpectralResponse

__all__ = [
    "UNMAPPED_MATERIAL_ID",
    "UNMAPPED_NAME",
    "MaterialTable",
    "StaleMaterialTableError",
    "THERMAL_COLUMNS",
    "ANGULAR_A_PLACEHOLDER",
    "ANGULAR_P_PLACEHOLDER",
    "CLOSURE_GUARD",
]

#: The cos θ grid the angle LUT is sampled on: uniform in **cos θ**, not in θ. A kernel has `n·v`
#: in hand and would have to call acos to index an angle-uniform table; uniform in cos θ makes the
#: lookup a multiply and a floor.
#:
#: The spacing that buys is uneven in angle, and in the unhelpful direction: one step of Δcos is
#: ~10° near normal, where ε is flat, and ~1° near grazing, where it is falling off a cliff. So
#: the node count is set by measurement rather than by taste -- at 33 nodes a Level A material
#: (water, whose ε runs to zero at grazing) interpolates 0.0048 from the exact dispatch, which is
#: a quarter of the 0.02 §4.2 allows Level B itself; at 65 it is 0.0012. Inside §4.2's own 70°
#: bound every material lands within 2e-4, two orders below that 0.02, so the packing is nowhere
#: near the limiting approximation in the chain. The whole table
#: is 19 x 65 float32.
ANGLE_LUT_COS: NDArray[np.float32] = np.linspace(1.0, 0.0, 65, dtype=np.float32)

UNMAPPED_MATERIAL_ID = 0
UNMAPPED_NAME = "UNMAPPED"
THERMAL_COLUMNS: tuple[str, ...] = (
    "density_kg_m3",
    "specific_heat_j_kgk",
    "conductivity_w_mk",
    "thickness_m",
    "solar_absorptivity",
    "heat_capacity_j_m2_k",
)
ANGULAR_A_PLACEHOLDER = 0.0  # Level C: no falloff until the Level-B fit (M7.7)
ANGULAR_P_PLACEHOLDER = 4.0
# Slack on the derived reflectance before it is called a closure violation (float32 packing).
CLOSURE_GUARD = 1e-6
TABLE_FORMAT = 1


class StaleMaterialTableError(RuntimeError):
    """The table on disk was packed from a different material library."""


@dataclass(frozen=True)
class MaterialTable:
    """Dense id → per-band columns for one band. Ids are small non-negative integers."""

    emissivity: NDArray[np.float32]  # index = material id; NaN where undefined
    band_id: str = ""
    reflectance: NDArray[np.float32] | None = None
    transmittance: NDArray[np.float32] | None = None
    angular_a: NDArray[np.float32] | None = None
    angular_p: NDArray[np.float32] | None = None
    roughness: NDArray[np.float32] | None = None
    #: (n_ids, n_angles) float32 ε(θ) for **every** material, sampled on :data:`ANGLE_LUT_COS`
    #: (M7.10). One table for all three §4.2 levels, so a kernel does a single lookup instead of
    #: branching on a material's angular model -- and so a Fresnel material, which needs an n/k
    #: file and a band average, can be evaluated on a GPU that has neither. The (a, p) columns
    #: stay because §4.2's "two instructions in a shader" is cheaper still where it applies.
    angle_lut: NDArray[np.float32] | None = None
    thermal: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    names: tuple[str, ...] = ()  # index = id; names[0] == UNMAPPED_NAME when set
    library_hash: str = ""

    def __post_init__(self) -> None:
        n = self.emissivity.shape[0]
        for name, arr in self._columns().items():
            if arr.dtype == np.float16:
                raise TypeError(f"{name} is float16 (CLAUDE.md #2): the table is float32")
            if arr.dtype != np.float32:
                raise TypeError(f"{name} must be float32, got {arr.dtype}")
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
        if self.angle_lut is not None:
            if self.angle_lut.dtype == np.float16:
                raise TypeError("angle_lut is float16 (CLAUDE.md #2): the table is float32")
            if self.angle_lut.dtype != np.float32:
                raise TypeError(f"angle_lut must be float32, got {self.angle_lut.dtype}")
            if self.angle_lut.shape != (n, ANGLE_LUT_COS.size):
                raise ValueError(
                    f"angle_lut has shape {self.angle_lut.shape}, expected "
                    f"({n}, {ANGLE_LUT_COS.size})"
                )
        if self.names and (len(self.names) != n or self.names[0] != UNMAPPED_NAME):
            raise ValueError("names must have one entry per id with names[0] == 'UNMAPPED'")

    def epsilon_at(self, material_id: Any, cos_theta: Any) -> NDArray[np.float32]:
        """ε(θ) by linear interpolation in the packed LUT -- what the kernel will do.

        Takes ǀcos θǀ, like every other angular path in this project: two sides of a thin panel
        have the same emissivity.
        """
        if self.angle_lut is None:
            raise ValueError("this table was packed without an angle LUT (M7.10)")
        ids = np.asarray(material_id)
        c = np.abs(np.asarray(cos_theta, dtype=np.float32))
        grid = ANGLE_LUT_COS[::-1].astype(np.float64)
        rows = self.angle_lut[ids][..., ::-1].astype(np.float64)
        position = np.interp(c.astype(np.float64), grid, np.arange(grid.size, dtype=np.float64))
        lo = np.clip(np.floor(position).astype(np.intp), 0, grid.size - 2)
        frac = position - lo
        taken_lo = np.take_along_axis(rows, lo[..., None], axis=-1)[..., 0]
        taken_hi = np.take_along_axis(rows, (lo + 1)[..., None], axis=-1)[..., 0]
        return np.asarray(taken_lo + frac * (taken_hi - taken_lo), dtype=np.float32)

    def _columns(self) -> dict[str, NDArray[np.float32]]:
        cols: dict[str, NDArray[np.float32]] = {"emissivity": self.emissivity}
        for key in ("reflectance", "transmittance", "angular_a", "angular_p", "roughness"):
            arr = getattr(self, key)
            if arr is not None:
                cols[key] = arr
        for key, arr in self.thermal.items():
            cols[f"thermal.{key}"] = arr
        return cols

    @property
    def n_ids(self) -> int:
        return int(self.emissivity.shape[0])

    @property
    def is_unmapped(self) -> NDArray[np.bool_]:
        flags = np.zeros(self.n_ids, dtype=bool)
        flags[UNMAPPED_MATERIAL_ID] = True
        return flags

    def id_for(self, name: str) -> int:
        if not self.names:
            raise ValueError("this table carries no names (built from a mapping)")
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"unknown material {name!r}; have {self.names[1:]}") from None

    def name_for(self, material_id: int) -> str:
        return self.names[material_id] if self.names else str(material_id)

    # -- construction from the library ------------------------------------------------
    @classmethod
    def from_library(
        cls,
        library: MaterialLibrary,
        band: str,
        response: SpectralResponse | None = None,
        form: WeightingForm = "energy",
        angle_lut: bool = False,
        data_dir: object = None,
    ) -> MaterialTable:
        """Pack every material of the library for one band; ids 1..N in sorted-name order.

        ``angle_lut=True`` additionally samples ε(θ) for every material through the M7.7 level
        dispatch (M7.10). Off by default because it costs a Fresnel band average per material and
        because every table packed before M7.10 must keep hashing the same.
        """
        from irsim.config.materials import EmpiricalAngular

        names = sorted(library.names)
        n = len(names) + 1
        eps = np.full(n, np.nan, dtype=np.float32)
        rho = np.full(n, np.nan, dtype=np.float32)
        tau = np.full(n, np.nan, dtype=np.float32)
        a = np.full(n, ANGULAR_A_PLACEHOLDER, dtype=np.float32)
        p_ = np.full(n, ANGULAR_P_PLACEHOLDER, dtype=np.float32)
        rough = np.full(n, np.nan, dtype=np.float32)
        # ``np.full(n, ...)`` is typed with a 1-element shape tuple under numpy 2.x while the
        # field is declared shape-agnostic; the annotation restates what the code already does.
        thermal: dict[str, NDArray[np.float32]] = {
            key: np.full(n, np.nan, dtype=np.float32) for key in THERMAL_COLUMNS
        }
        for i, name in enumerate(names, start=1):
            m = library[name]
            props = m.band_properties(band, response, form)
            eps[i], rho[i], tau[i] = props.emissivity, props.reflectance, props.transmittance
            angular = m.spec.optical.angular_model
            if isinstance(angular, EmpiricalAngular):
                a[i], p_[i] = angular.a, angular.p
            rough[i] = (m.spec.optical.roughness_per_band or {}).get(band, np.nan)
            t = m.spec.thermal
            for key in THERMAL_COLUMNS:
                thermal[key][i] = getattr(t, key)
        lut = None
        if angle_lut:
            from irsim.materials.directional import directional_emissivity

            lut = np.full((n, ANGLE_LUT_COS.size), np.nan, dtype=np.float32)
            for i, name in enumerate(names, start=1):
                lut[i] = directional_emissivity(
                    library[name], band, ANGLE_LUT_COS, response, form=form, data_dir=data_dir
                )
        return cls(
            emissivity=eps,
            band_id=band,
            angle_lut=lut,
            reflectance=rho,
            transmittance=tau,
            angular_a=a,
            angular_p=p_,
            roughness=rough,
            thermal=thermal,
            names=(UNMAPPED_NAME, *names),
            library_hash=library.content_hash(),
        )

    @classmethod
    def for_sensor(
        cls,
        library: MaterialLibrary,
        sensor: SensorSpec | SensorConfig,
        angle_lut: bool = False,
        data_dir: object = None,
    ) -> MaterialTable:
        """Pack the library for **this camera**: its band, its R(λ), its detector's weighting.

        :meth:`from_library` with only a band name averages every curve over the band's nominal
        top-hat in energy form -- the right number for a camera that does not exist. A 6-13 µm
        camera and a 7.5-13.5 µm one are both ``lwir``, and only the response tells them apart
        (AT.33, ADR 0175); a photon FPA counts photons, so it averages under B_q (ADR 0021). A
        material without a curve packs the same value either way.
        """
        from irsim.config.loader import resolve_data_dir
        from irsim.config.sensor import SensorConfig as _SensorConfig
        from irsim.materials.spectra import weighting_for_fpa
        from irsim.radiometry.band import Band

        spec = sensor.sensor if isinstance(sensor, _SensorConfig) else sensor
        # The same resolution and edge check the LUT builder applies, so table and LUT describe
        # one camera.
        path = pathlib.Path(spec.band.spectral_response)
        if not path.is_absolute():
            path = resolve_data_dir(data_dir) / path  # type: ignore[arg-type]
        response = Band.from_spec(spec.band, path).response
        form = weighting_for_fpa(spec.fpa.type)
        return cls.from_library(library, spec.band.band_id, response, form, angle_lut, data_dir)

    # -- files ----------------------------------------------------------------------------
    def save(self, path: str | os.PathLike[str]) -> tuple[pathlib.Path, pathlib.Path]:
        """Write ``<path>.npz`` + ``<path>.json`` sidecar (band, names, library hash)."""
        base = pathlib.Path(path)
        npz = base.with_suffix(".npz")
        side = base.with_suffix(".json")
        columns: dict[str, Any] = {k.replace(".", "__"): v for k, v in self._columns().items()}
        if self.angle_lut is not None:
            columns["angle_lut"] = self.angle_lut
        np.savez(str(npz), **columns)
        side.write_text(
            json.dumps(
                {
                    "format": TABLE_FORMAT,
                    "band_id": self.band_id,
                    "names": list(self.names),
                    "library_hash": self.library_hash,
                    "columns": sorted(self._columns()),
                    "angle_lut": None
                    if self.angle_lut is None
                    else {"n_angles": int(self.angle_lut.shape[1])},
                },
                indent=2,
            )
        )
        return npz, side

    @classmethod
    def load(
        cls, path: str | os.PathLike[str], library: MaterialLibrary | None = None
    ) -> MaterialTable:
        """Read a table; with ``library`` given, refuse one packed from a different library."""
        base = pathlib.Path(path)
        meta: dict[str, Any] = json.loads(base.with_suffix(".json").read_text())
        if meta.get("format") != TABLE_FORMAT:
            raise ValueError(f"material table format {meta.get('format')} != {TABLE_FORMAT}")
        if library is not None and meta["library_hash"] != library.content_hash():
            raise StaleMaterialTableError(
                f"{base}: packed from library {meta['library_hash'][:12]}, current library is "
                f"{library.content_hash()[:12]} -- regenerate the table"
            )
        with np.load(base.with_suffix(".npz")) as data:
            arrays = {k.replace("__", "."): np.asarray(data[k]) for k in data.files}
        for arr in arrays.values():
            if arr.dtype == np.float16:
                raise TypeError("material table on disk is float16 (CLAUDE.md #2)")
        thermal = {
            k.split(".", 1)[1]: arrays[k].astype(np.float32)
            for k in arrays
            if k.startswith("thermal.")
        }
        return cls(
            emissivity=arrays["emissivity"].astype(np.float32),
            band_id=str(meta["band_id"]),
            reflectance=arrays.get("reflectance"),
            transmittance=arrays.get("transmittance"),
            angular_a=arrays.get("angular_a"),
            angular_p=arrays.get("angular_p"),
            roughness=arrays.get("roughness"),
            angle_lut=arrays.get("angle_lut"),
            thermal=thermal,
            names=tuple(meta["names"]),
            library_hash=str(meta["library_hash"]),
        )

    @classmethod
    def from_mapping(cls, eps_by_id: Mapping[int, float], band_id: str = "") -> MaterialTable:
        if not eps_by_id:
            raise ValueError("material table needs at least one material")
        if UNMAPPED_MATERIAL_ID in eps_by_id:
            raise ValueError(
                "material id 0 is the UNMAPPED sentinel and cannot carry an emissivity"
            )
        max_id = max(eps_by_id)
        if min(eps_by_id) < 0 or max_id > 65535:
            raise ValueError("material ids must lie in 1..65535")
        table = np.full(max_id + 1, np.nan, dtype=np.float32)
        for mid, eps in eps_by_id.items():
            if not 0.0 < eps <= 1.0:
                raise ValueError(f"material {mid}: emissivity {eps} must lie in (0, 1]")
            table[mid] = np.float32(eps)
        return cls(emissivity=table, band_id=band_id)

    @classmethod
    def constant(cls, eps: float, ids: tuple[int, ...] = (1,), band_id: str = "") -> MaterialTable:
        return cls.from_mapping(dict.fromkeys(ids, eps), band_id)

    def properties_for(
        self, material_id: NDArray[np.integer], sky_mask: NDArray[np.bool_] | None = None
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Per-pixel (ε₀, ρ, τ) closing to 1 by construction (§4.1, §4.4).

        ε comes from :meth:`emissivity_for`, τ from the packed transmittance column (0 when the
        table has none, e.g. one built by :meth:`from_mapping`), and **ρ is derived** as
        1 − ε − τ rather than read back, so the closure cannot drift from the authored values
        (CLAUDE.md #4). Pixels under ``sky_mask`` are blackbody-equivalent: ε = 1, ρ = τ = 0.
        """
        eps = self.emissivity_for(material_id, sky_mask)
        ids = np.asarray(material_id)
        if self.transmittance is None:
            tau = np.zeros(eps.shape, dtype=np.float32)
        else:
            tau = np.asarray(self.transmittance[ids], dtype=np.float32)
            if sky_mask is not None:
                tau = np.where(np.asarray(sky_mask), np.float32(0.0), tau).astype(np.float32)
            if np.any(np.isnan(tau)):
                raise ValueError("material table has NaN transmittance for a referenced id")
        rho = np.asarray(1.0 - eps.astype(np.float64) - tau.astype(np.float64), dtype=np.float32)
        if np.any(rho < -CLOSURE_GUARD):
            worst = float(rho.min())
            raise ValueError(
                f"derived reflectance {worst:.6f} < 0: ε + τ > 1 for some material "
                "(the library loader should have refused it, CLAUDE.md #4)"
            )
        return eps, np.maximum(rho, np.float32(0.0)), tau

    def directional_properties_for(
        self,
        material_id: NDArray[np.integer],
        cos_theta: Any,
        sky_mask: NDArray[np.bool_] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Per-pixel (ε(θ), ρ, τ) from the packed angle LUT, closing to 1 at every angle.

        **ρ is re-derived at each angle, never carried.** ε(θ) moves and τ does not, so ρ must
        absorb the difference or the pixel stops conserving energy the moment the surface tilts --
        a per-pixel closure failure that grows towards the limb, which is exactly where a
        directional model is supposed to be improving things.

        τ is held angle-independent, which is an approximation and a stated one: a real
        transmittance falls towards grazing too. It is first order in the same sense §4.2's Level
        B is, and the materials with τ > 0 in this library (glass in SWIR, a leaf in NIR) are
        looked at near normal far more often than at the limb.

        Pixels under ``sky_mask`` keep the blackbody-equivalent ε = 1, ρ = τ = 0 of
        :meth:`properties_for`: the G-buffer carries the *apparent* sky temperature there, which
        is a radiance dressed as a temperature and has no surface to have an angle to.
        """
        if self.angle_lut is None:
            raise ValueError(
                "this table was packed without an angle LUT; pass angle_lut=True to "
                "MaterialTable.from_library (M7.10) or use properties_for for the ε₀ path"
            )
        ids = np.asarray(material_id)
        eps0, _rho0, tau = self.properties_for(ids, sky_mask)
        eps = self.epsilon_at(ids, cos_theta)
        if sky_mask is not None:
            eps = np.where(np.asarray(sky_mask), np.float32(1.0), eps).astype(np.float32)
        if np.any(np.isnan(eps)):
            raise ValueError("angle LUT has NaN for a referenced material id")
        headroom = 1.0 - tau.astype(np.float64)
        eps = np.asarray(np.minimum(eps.astype(np.float64), headroom), dtype=np.float32)
        rho = np.asarray(headroom - eps.astype(np.float64), dtype=np.float32)
        del eps0
        return eps, np.maximum(rho, np.float32(0.0)), tau

    def emissivity_for(
        self, material_id: NDArray[np.integer], sky_mask: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float32]:
        """Per-pixel ε₀; raises on the UNMAPPED sentinel or an id the table does not define.

        Pixels under ``sky_mask`` are blackbody-equivalent (ε₀ = 1: the G-buffer carries the
        *apparent* sky temperature there, see irsim.config.gbuffer) and their ids are not
        checked, so the renderer's background id 0 is not mistaken for an unmapped asset.
        """
        ids = np.asarray(material_id)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"material_id must be an integer plane, got {ids.dtype}")
        if sky_mask is not None:
            sky = np.asarray(sky_mask)
            if sky.dtype != np.bool_ or sky.shape != ids.shape:
                raise ValueError("sky_mask must be a bool plane with the material_id shape")
            eps = np.ones(ids.shape, dtype=np.float32)
            if np.any(~sky):
                eps[~sky] = self.emissivity_for(ids[~sky])
            return eps
        if np.any(ids == UNMAPPED_MATERIAL_ID):
            raise ValueError(
                "G-buffer contains material id 0 (UNMAPPED): an asset has no material mapping; "
                "fix the resolver rather than rendering a default emissivity"
            )
        if np.any(ids < 0) or np.any(ids >= self.emissivity.size):
            raise ValueError(f"material id outside the table (0..{self.emissivity.size - 1})")
        # A distinct name from the sky-mask branch above: the two have different inferred shape
        # types under numpy 2.x, and reusing one name makes the function's type depend on which
        # branch mypy saw first.
        looked_up = np.asarray(self.emissivity[ids], dtype=np.float32)
        if np.any(np.isnan(looked_up)):
            bad = sorted(set(np.unique(ids[np.isnan(looked_up)]).tolist()))
            raise ValueError(f"material ids {bad} have no emissivity in this table")
        return looked_up
