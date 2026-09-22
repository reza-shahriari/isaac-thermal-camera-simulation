"""Pipeline scaffolding: the stage protocol, the immutable configuration, the mutable state.

A ``Stage`` is a pure function of (planes, config, state) → planes. Planes are the G-buffer dict
contract (:mod:`irsim.config.gbuffer`) extended by the stages' own outputs (``radiance`` at the
supersampled grid, ``flux`` at the detector grid, ``dn16``...). Everything that carries
temperature or radiance is float32 or better (non-negotiable #2).

docs/physics-model.md §13.4, §13.6
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.model import Atmosphere
from irsim.atmosphere.sky import SkyModel
from irsim.config.loader import load_sensor_config
from irsim.config.sensor import SensorConfig
from irsim.detector.anchor import anchor_noise
from irsim.detector.bolometer import MicrobolometerDetector
from irsim.detector.cold_shield import background_electrons
from irsim.detector.params import BolometerParams, FpaParams, PhotonParams, fpa_params_from_config
from irsim.detector.photon import PhotonDetector
from irsim.detector.response import Detector
from irsim.isp.radiometric import RadiometricCalibration
from irsim.materials.table import MaterialTable
from irsim.noise.electron import electron_budget
from irsim.noise.stage import NoiseStage
from irsim.optics.psf import optical_psf
from irsim.radiometry.lut import BandLUT, Quantity
from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = ["Planes", "Stage", "PipelineConfig", "PipelineState", "RADIOMETRIC_RANGE_K"]

Planes = dict[str, NDArray[Any]]

# Scene-temperature span that fills the ADC for the calibrated transfer (ADR 0021): -40..+200 C.
RADIOMETRIC_RANGE_K: tuple[float, float] = (233.15, 473.15)


@dataclass(frozen=True)
class PipelineConfig:
    """Everything a frame needs that does not change between frames."""

    sensor: SensorConfig
    lut: BandLUT
    materials: MaterialTable
    fpa: FpaParams
    supersample: int
    calibration: RadiometricCalibration | None
    t_housing_cal_k: float
    detector: Detector
    noise: NoiseStage
    sensor_seed: int
    noise_enabled: bool
    psf: NDArray[np.float64] | None  # optical PSF at the k× pitch; None = no optical blur
    atmosphere: Atmosphere | LayeredAtmosphere | None = None  # stage 2; None = identity
    tau_override: float | None = (
        None  # L1 fallback: constant τ, path radiance at the weather's T_air
    )
    sky: SkyModel | None = None  # stage-1 reflected term (M7.13); None = emission only
    #: The camera's own flat-field correction, applied in the **display branch only** (M9.12).
    #: `dn16` and the radiometric outputs stay the raw ADC plane: the radiometric branch already
    #: divides cos⁴ out analytically in `invert_optics`, so correcting it again would remove the
    #: same term twice. ``None`` is the un-flat-fielded camera, which is what every golden written
    #: before M9.12 describes.
    flat_field: Any = None  # TwoPointNuc; Any avoids a cycle through irsim.isp
    #: The M9 sensor chain (M9.8): housing and FPA nodes, pattern drift, defects, NUC
    #: residual and the FFC. ``None`` is the ideal camera -- the chain every M9 mechanism is
    #: measured against -- and is the default so that existing benches and goldens describe
    #: the radiometry alone. ``attach_sensor_chain`` turns it on.
    chain: Any = None  # SensorChain; Any avoids a cycle through irsim.pipeline.sensor_chain
    #: The camera's own R(λ), kept because stage 2d integrates over it: a plume's band radiance
    #: and soot's band-mean κ are quadratures, not LUT lookups, since the LUT stops at 1000 K and
    #: a flame does not (`PH.6`).
    response: SpectralResponse | None = None
    #: The band's hot-gas absorption tables (`PH.5`). ``None`` for a band the model does not
    #: reach — SWIR and NIR on the committed RadCal tables — and a plume carrying gas in such a
    #: band raises in stage 2d rather than rendering as a clear one.
    gas_tables: Any = None  # GasBandTables; Any avoids a cycle through irsim.pipeline.gas_slab

    @property
    def quantity(self) -> Quantity:
        """Which LUT table the radiance chain runs on (ADR 0021).

        Delegates to the sensor config so a caller that has only a ``SensorConfig`` -- a render
        script building its ``Scene`` before the pipeline exists, say -- reaches the same answer
        and cannot build the scene in the other form.
        """
        return self.sensor.sensor.quantity

    @classmethod
    def from_sensor(
        cls,
        sensor: SensorConfig,
        materials: MaterialTable,
        lut: BandLUT | None = None,
        lut_dir: str | os.PathLike[str] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
        t_housing_cal_k: float | None = None,
        radiometric_range_k: tuple[float, float] = RADIOMETRIC_RANGE_K,
        sensor_seed: int = 0,
        noise_enabled: bool | None = None,
        psf_enabled: bool | None = None,
        reference_wavelength_um: float | None = None,
        atmosphere: Atmosphere | LayeredAtmosphere | None = None,
        tau_override: float | None = None,
        sky: SkyModel | None = None,
        flat_field_enabled: bool = False,
        noise_handle: Literal["auto", "netd", "electrons"] = "auto",
    ) -> PipelineConfig:
        """Assemble from a validated sensor config; the LUT is given or loaded from ``lut_dir``.

        The calibrated transfer (ADR 0021) is built for bolometer cameras with the housing at
        ``t_housing_cal_k`` (default: ``optics.housing_temp_k`` if fixed, else 300 K). The
        detector's noise budget is built from whichever handle its datasheet actually offers
        (``noise_handle``, below) and the correlated 3-D stage seeded with ``sensor_seed``
        (ADR 0022); ``noise_enabled=False`` runs the ideal chain (the ablation switch of ME.8).
        ``noise_handle='netd'`` forces the ADR 0025 anchor for a photon FPA too, which is the
        ablation the M11.6 addendum argues against but does not forbid.
        The optical PSF (diffraction at the band-
        representative wavelength -- ``reference_wavelength_um``, else
        ``mtf.reference_wavelength_um``, else the band centre -- times the aberration Gaussian) is
        built at the supersampled pitch (ADR 0059); ``psf_enabled=False`` skips it. Stage 2
        runs when an ``Atmosphere`` (M8.5, bound to the scene's WeatherSeries) is given; the
        frame time on the weather axis is ``PipelineState.t_s``. ``tau_override`` is the L1
        fallback: a constant τ at every distance, path radiance still at the weather's T_air.
        """
        fidelity = sensor.sensor.fidelity
        noise_enabled = fidelity.noise if noise_enabled is None else noise_enabled
        psf_enabled = fidelity.optical_psf if psf_enabled is None else psf_enabled
        if tau_override is not None:
            if atmosphere is None:
                raise ValueError("tau_override needs an Atmosphere (its weather gives T_air)")
            if not 0.0 <= tau_override <= 1.0:
                raise ValueError("tau_override must lie in [0, 1]")
        if atmosphere is not None and sensor.sensor.band.band_id not in atmosphere.preset.bands:
            raise ValueError(
                f"atmosphere preset has no band {sensor.sensor.band.band_id!r} "
                f"(has {sorted(atmosphere.preset.bands)})"
            )
        if tau_override is not None and isinstance(atmosphere, LayeredAtmosphere):
            raise ValueError(
                "tau_override is the grey L1 fallback; use the grey Atmosphere with it"
            )
        if sky is not None:
            if sky.band != sensor.sensor.band.band_id:
                raise ValueError(
                    f"sky model is for band {sky.band!r}, sensor is {sensor.sensor.band.band_id!r}"
                )
            if atmosphere is not None and sky.weather is not atmosphere.weather:
                raise ValueError("sky model and atmosphere hold different WeatherSeries (#6)")
            if sky.environment.ground.mode == "solver":
                raise ValueError("ground.mode 'solver' needs the environment solver (M6.12)")
            expected_q = "lb" if fpa_params_from_config(sensor).type == "bolometer" else "lb_q"
            if sky.quantity != expected_q:
                raise ValueError(
                    f"sky model built in the {sky.quantity!r} form; "
                    f"this sensor runs on {expected_q!r}"
                )
        if lut is None:
            if lut_dir is None:
                raise ValueError("give a BandLUT or a lut_dir to load one from (make luts)")
            lut = load_band_lut_for_config(sensor, lut_dir, data_dir)
        fpa = fpa_params_from_config(sensor)
        if t_housing_cal_k is None:
            fixed = sensor.sensor.optics.housing_temp_k
            t_housing_cal_k = fixed if fixed is not None else 300.0
        calibration = None
        # §9.1's cold shield (M11.5, ADR 0066): a mismatched shield opens a wider cone than the
        # lens fills, and the difference is warm dewar structure. Exactly zero at eta_cs = 1, so
        # every uncooled camera in this repository is unaffected -- and it is the *shot noise* of
        # this offset that costs NETD, not the offset itself, which NUC removes.
        background = 0.0
        eta_cs = sensor.sensor.optics.cold_shield_efficiency
        if isinstance(fpa, PhotonParams) and eta_cs < 1.0:
            background = background_electrons(
                float(lut.lookup(np.float64(t_housing_cal_k), "lb_q")[()]),
                fpa,
                sensor.sensor.optics.f_number,
                eta_cs,
            )
        # §9.4 / §10.1: which handle sets the magnitude of the noise (ADR 0025 and its M11.6
        # addendum). A bolometer's datasheet quotes NETD and nothing else usable, so it is solved
        # for. A photon FPA's quotes the thing the noise is *made of* -- quantum efficiency, well,
        # integration time, read noise in electrons, dark current -- so those are used directly
        # and NETD becomes the cross-check. "auto" picks the second whenever the config carries
        # the electron numbers, which is what SC.1 wires up: until now every photon camera in this
        # repository was anchored, which overstated the MWIR InSb's read noise by 1.52x (533 e-
        # against its datasheet 350) and rendered the SWIR InGaAs with dark = 0 against its own
        # config's 200 e- per integration.
        authored = isinstance(fpa, PhotonParams) and fpa.read_noise_e is not None
        use_electrons = {"auto": authored, "electrons": True, "netd": False}[noise_handle]
        if use_electrons:
            if not isinstance(fpa, PhotonParams):
                raise ValueError(
                    "noise_handle='electrons' is for a photon FPA; a bolometer has no electron "
                    "datasheet to build from and keeps the ADR 0025 NETD anchor"
                )
            budget = electron_budget(sensor.sensor, lut, background_electrons=background)
        else:
            budget = anchor_noise(sensor.sensor, lut, background_electrons=background)
        detector: Detector
        if isinstance(fpa, BolometerParams):
            calibration = RadiometricCalibration.from_scene_range(
                sensor.sensor, lut, radiometric_range_k[0], radiometric_range_k[1], t_housing_cal_k
            )
            detector = MicrobolometerDetector(fpa, calibration.transfer, budget)
        elif isinstance(fpa, PhotonParams):
            detector = PhotonDetector(fpa, budget)
        else:  # pragma: no cover
            raise TypeError(f"unknown FPA params {type(fpa).__name__}")
        spec = sensor.sensor
        psf = None
        if psf_enabled:
            lam = (
                reference_wavelength_um
                if reference_wavelength_um is not None
                else spec.reference_wavelength_um
            )
            psf = optical_psf(
                lam,
                spec.optics.f_number,
                spec.optics.mtf.aberration_sigma_um,
                spec.fpa.pitch_um,
                spec.optics.supersample_factor,
            )
        # Stage 2d's two inputs (`PH.6`). Neither is fatal to build without: a camera with no
        # response file on disk still renders everything but a plume, and a band the absorption
        # model does not reach (SWIR, NIR) gets `None` here and a clear error there rather than
        # a silent zero.
        try:
            response = load_band_response_for_config(sensor, data_dir)
        except (FileNotFoundError, ValueError):
            response = None
        gas_tables = None
        if response is not None:
            from irsim.pipeline.gas_tables import gas_tables_for

            try:
                gas_tables = gas_tables_for(spec.band.band_id, response, data_dir)
            except (FileNotFoundError, ValueError):
                gas_tables = None
        built = cls(
            sensor=sensor,
            lut=lut,
            materials=materials,
            fpa=fpa,
            supersample=spec.optics.supersample_factor,
            calibration=calibration,
            t_housing_cal_k=t_housing_cal_k,
            detector=detector,
            noise=NoiseStage.from_sensor(spec, sensor_seed, enabled=noise_enabled),
            sensor_seed=sensor_seed,
            noise_enabled=noise_enabled,
            psf=psf,
            atmosphere=atmosphere,
            tau_override=tau_override,
            sky=sky,
            response=response,
            gas_tables=gas_tables,
        )
        if flat_field_enabled:
            from irsim.pipeline.flat_field import calibrate_flat_field

            built = replace(built, flat_field=calibrate_flat_field(built))
        return built

    @classmethod
    def from_yaml(
        cls,
        path: str | os.PathLike[str],
        materials: MaterialTable,
        lut_dir: str | os.PathLike[str],
        data_dir: str | os.PathLike[str] | None = None,
    ) -> PipelineConfig:
        return cls.from_sensor(
            load_sensor_config(path, data_dir), materials, None, lut_dir, data_dir
        )


@dataclass
class PipelineState:
    """Per-camera state that persists across frames: frame counter, housing temperature,
    and the buffers later stages keep (bolometer IIR, drift, FFC reference)."""

    frame_index: int = 0
    housing_temp_k: float = 300.0
    t_s: float = (
        0.0  # frame time on the scene weather's axis (Scene.t0_s + t_rel); stage 2 reads it
    )
    #: Cross-frame buffers, one owner and one reset path each (ADR 0052). NumPy arrays on the
    #: reference path; the Warp path (M10.6) keeps its device-resident equivalents here too, so
    #: the values are not all ndarrays and moving a stage to the GPU stays a transport change.
    buffers: dict[str, Any] = field(default_factory=dict)

    def advance(self) -> None:
        self.frame_index += 1


class Stage(Protocol):
    """One pipeline stage: reads the planes it needs, returns the planes it produces."""

    name: str

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes: ...


def require_fp32_or_better(plane: NDArray[Any], name: str) -> NDArray[np.floating]:
    if plane.dtype == np.float16:
        raise TypeError(f"{name} is float16 (CLAUDE.md non-negotiable #2); use float32 or better")
    if not np.issubdtype(plane.dtype, np.floating):
        raise TypeError(f"{name} must be a float plane, got {plane.dtype}")
    return plane
