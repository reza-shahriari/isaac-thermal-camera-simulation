"""Sensor configuration schema: the typed, validated form of a §12.2 camera file.

A band is a data file (docs/physics-model.md §12). This module is the contract for that file:
every field of §12.2 with units in its name, every ``a | b | c`` comment a ``Literal``, and the
physical constraints a value must satisfy checked here, once, so nothing downstream re-validates.
Models are frozen and reject unknown keys (a typo such as ``f_stop`` must fail, not default).

Deliberately absent: an aperture-factor property. The factor π/(4F² + 1) is defined once in
``irsim.optics`` (CLAUDE.md non-negotiable #5); the config exposes ``f_number`` only.

Loading YAML, resolving data paths and hashing are the loader's job (``irsim.config.loader``);
this module builds from plain dictionaries.

docs/physics-model.md §12.1, §12.2, §16.1; §8.1, §8.3, §9.1-§9.4, §10.2, §11.2-§11.4 for the
meaning of the individual fields.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from irsim.config.bands import BandId, band_id_for
from irsim.radiometry.constants import WAVELENGTH_MAX_UM, WAVELENGTH_MIN_UM

__all__ = [
    "SCHEMA_VERSION",
    "MIN_SCHEMA_VERSION",
    "FidelitySpec",
    "FULL_FIDELITY",
    "SensorConfig",
    "SensorSpec",
    "BandSpec",
    "OpticsSpec",
    "MtfSpec",
    "DistortionSpec",
    "BolometerFpa",
    "PhotonFpa",
    "DarkCurrentSpec",
    "FpaSpec",
    "NoiseSpec",
    "Ratios3D",
    "RATIO_ORDER",
    "BadPixelTypeMix",
    "NucSpec",
    "FpaTempMode",
    "IspSpec",
    "FocusSpec",
    "OutputsSpec",
]

# 2: optional optics fields (housing, supersample, mtf, vignetting_map); 3: optional detector
# constants (bolometer thermal/bias, photon dark current & read noise, FPA thermal node);
# 4: noise.bad_pixel_type_mix and noise.netd_ref_f_number. ADR 0017.
# 5: fpa_temp_mode and the raw gain/offset(T_FPA) polynomials with their T_cal (M9.2, ADR 0053).
# 6: housing_temp_mode 'coupled' now requires housing_tau_s -- a tightening, not a new field,
# because M9.3 gave the coupled housing an integrator and it needs a time constant to be one.
# 7: noise.bad_pixel_rts_{occupancy,dwell_frames,amplitude_dn} for the §10.4 flickering and
# blinking classes (M9.5a, ADR 0055). All three default, so existing configs are unchanged.
# 8: nuc.shutterless_tau_s, the scene-based-correction time constant that bounds a shutterless
# core's residual (M9.7, ADR 0057). Defaults, and is unused outside `mode: shutterless`.
# 9: the optional `fidelity:` block (ME.8).
# 10: `optics.focus` and `optics.mtf.defocus_model`/`defocus_apply` (`OC.4`, ADR 0129). Every
# default is the pre-v10 camera -- focused at infinity with no defocus model -- so a v9 document is
# a valid v10 document describing exactly the camera it described before.
SCHEMA_VERSION = 10
#: The oldest version this loader still accepts. v9 added `fidelity:` as an **optional** block whose
#: default is full fidelity, so every v8 document is a valid v9 document and describes exactly the
#: camera it described before. A range is the honest representation of a backwards-compatible
#: change; raise this floor only when a version genuinely stops being readable.
MIN_SCHEMA_VERSION = 8

Regime = Literal["emissive", "reflective", "mixed"]
HousingTempMode = Literal["fixed", "ambient", "coupled"]
# The FPA node takes the same three modes as the housing node; 'fixed' is a TEC-pinned core.
FpaTempMode = Literal["fixed", "ambient", "coupled"]
DistortionModel = Literal["brown_conrady", "kannala_brandt", "ftheta"]
NucMode = Literal["shuttered", "shutterless", "ideal"]
AgcMode = Literal["linear", "plateau_equalization", "plateau_local", "none"]
Polarity = Literal["white_hot", "black_hot"]
# §11.4 lists ironbow/rainbow/lava/arctic and §12.2 gray/ironbow/rainbow/lava: both accepted (S27).
Palette = Literal["gray", "ironbow", "rainbow", "lava", "arctic"]
# `OC.4`: how the lens is focused, and which model turns the resulting W020 into an OTF.
# `infinity` + `none` is what every configuration written before v10 describes.
FocusMode = Literal["infinity", "hyperfocal", "fixed"]
DefocusModel = Literal["none", "gaussian", "geometric", "hopkins"]
DefocusApply = Literal["global", "layered"]

# Regime-vs-wavelength consistency (§12.1): self-emission at scene temperatures is negligible
# below ~2.5 µm and reflected sunlight is negligible beyond ~3 µm.
EMISSIVE_MIN_LAMBDA_MAX_UM = 2.5
REFLECTIVE_MAX_LAMBDA_MIN_UM = 3.0


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BandSpec(_Frozen):
    """§12.2 ``band``. ``spectral_response`` is a path relative to the data root; the loader
    resolves it and the response file contract lives in ``irsim.radiometry`` (M1.3).

    ``id`` is optional: when absent it is derived from the edges by :func:`band_id_for`; when
    present it must agree with that derivation (a ``mwir`` label on a 7.5-13.5 um band is a
    config error, not a preference). Read :attr:`band_id` for the resolved value."""

    lambda_min_um: float = Field(gt=0)
    lambda_max_um: float = Field(gt=0)
    spectral_response: str
    regime: Regime
    id: BandId | None = None

    @property
    def band_id(self) -> BandId:
        return (
            self.id if self.id is not None else band_id_for(self.lambda_min_um, self.lambda_max_um)
        )

    @model_validator(mode="after")
    def _physical(self) -> BandSpec:
        lo, hi = self.lambda_min_um, self.lambda_max_um
        if not (WAVELENGTH_MIN_UM <= lo < hi <= WAVELENGTH_MAX_UM):
            raise ValueError(
                f"band [{lo}, {hi}] um must satisfy {WAVELENGTH_MIN_UM} <= min < max <= "
                f"{WAVELENGTH_MAX_UM} -- wavelengths are MICROMETRES (7500 means nanometres)"
            )
        if self.regime == "emissive" and hi < EMISSIVE_MIN_LAMBDA_MAX_UM:
            raise ValueError(
                f"regime 'emissive' with lambda_max {hi} um: self-emission at scene temperatures "
                f"is negligible below {EMISSIVE_MIN_LAMBDA_MAX_UM} um (§12.1)"
            )
        if self.regime == "reflective" and lo > REFLECTIVE_MAX_LAMBDA_MIN_UM:
            raise ValueError(
                f"regime 'reflective' with lambda_min {lo} um: reflected sunlight is negligible "
                f"beyond {REFLECTIVE_MAX_LAMBDA_MIN_UM} um; use 'mixed' or 'emissive' (§12.1)"
            )
        derived = band_id_for(lo, hi)  # raises if the edges match no canonical band
        if self.id is not None and self.id != derived:
            raise ValueError(
                f"band.id {self.id!r} contradicts the edges [{lo}, {hi}] um, which are {derived!r}"
            )
        return self


class DistortionSpec(_Frozen):
    """§12.2 ``optics.distortion``; applied by the engine, stored here as schema (ADR 0015)."""

    model: DistortionModel
    coeffs: list[float]

    @model_validator(mode="after")
    def _coeff_count(self) -> DistortionSpec:
        expected = {"brown_conrady": 5, "kannala_brandt": 4}.get(self.model)
        if expected is not None and len(self.coeffs) != expected:
            raise ValueError(f"{self.model} takes {expected} coefficients, got {len(self.coeffs)}")
        if self.model == "ftheta" and not self.coeffs:
            raise ValueError("ftheta needs at least one coefficient")
        return self


class FocusSpec(_Frozen):
    """Where the lens is focused (`OC.4`; docs/physics-model.md §8.3 names no such field).

    ``infinity`` is the default and is what every camera in this repository described before v10:
    a collimated-target lab setup, and a fair model of a lens focused past its hyperfocal.

    ``hyperfocal`` is the fixed-focus core most Bosons actually are. It needs an **acceptable**
    circle of confusion, which is a convention and not a measurement -- one detector pitch is the
    usual choice for a sampled imager, and ``coc_um`` defaults to the pitch rather than being
    silently assumed somewhere downstream.

    ``fixed`` is a lens focused once at a known range and needs ``distance_m``.
    """

    mode: FocusMode = "infinity"
    distance_m: float | None = Field(default=None, gt=0)
    coc_um: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _consistency(self) -> FocusSpec:
        if self.mode == "fixed" and self.distance_m is None:
            raise ValueError("focus mode 'fixed' requires distance_m")
        if self.mode != "fixed" and self.distance_m is not None:
            raise ValueError(f"focus mode {self.mode!r} does not take distance_m")
        if self.mode != "hyperfocal" and self.coc_um is not None:
            raise ValueError("coc_um is only meaningful for focus mode 'hyperfocal'")
        return self


class MtfSpec(_Frozen):
    """Optional MTF parameters the spec leaves open (§8.3; ADR 0017 schema-gap policy).

    ``aberration_sigma_um`` is the Gaussian fitted from a measured slant edge (0 = none);
    ``reference_wavelength_um`` sets the diffraction cut-off ξ_c = 1/(λF) and defaults to the
    band centre; ``apply_motion_mtf`` enables the image-plane motion term for photon FPAs.
    """

    aberration_sigma_um: float = Field(default=0.0, ge=0)
    reference_wavelength_um: float | None = Field(default=None, gt=0)
    apply_motion_mtf: bool = False
    #: `OC.4`, ADR 0129. `hopkins` carries diffraction and defocus in one term and is the only one
    #: valid in the transition band this project works in; `geometric` and `gaussian` are ablations.
    #: `none` is the pre-v10 camera -- perfectly focused at every range.
    defocus_model: DefocusModel = "none"
    #: `global` is one kernel for the frame (`OC.5`); `layered` buckets by depth (`OC.6`).
    defocus_apply: DefocusApply = "global"


RECTILINEAR_MODELS = frozenset({"brown_conrady"})
SUPERSAMPLE_MIN, SUPERSAMPLE_MAX = 1, 8


class OpticsSpec(_Frozen):
    """§12.2 ``optics`` plus the optional fields the spec omits (ADR 0017).

    No aperture factor here -- see ``irsim.optics`` (non-negotiable #5). The housing fields feed
    the self-emission term (ADR 0016): ``fixed`` needs ``housing_temp_k``; ``coupled`` needs the
    lag ``housing_tau_s`` and may set the steady self-heating ``housing_self_heating_k`` above
    ambient. ``irsim.optics.HousingTemperature`` (M9.3) is what reads these.
    """

    f_number: float = Field(gt=0)
    focal_length_mm: float = Field(gt=0)
    transmittance: float = Field(gt=0, le=1)
    housing_temp_mode: HousingTempMode
    cold_shield_efficiency: float = Field(ge=0, le=1)
    distortion: DistortionSpec
    vignetting_cos4: bool
    housing_temp_k: float | None = Field(default=None, gt=0)
    housing_tau_s: float | None = Field(default=None, gt=0)
    housing_self_heating_k: float = Field(default=0.0, ge=0)
    vignetting_map: str | None = None
    supersample_factor: int = Field(default=4, ge=SUPERSAMPLE_MIN, le=SUPERSAMPLE_MAX)
    mtf: MtfSpec = Field(default_factory=MtfSpec)
    focus: FocusSpec = Field(default_factory=FocusSpec)

    @model_validator(mode="after")
    def _consistency(self) -> OpticsSpec:
        if self.housing_temp_mode == "fixed" and self.housing_temp_k is None:
            raise ValueError("housing_temp_mode 'fixed' requires housing_temp_k")
        if self.housing_temp_mode == "coupled" and self.housing_tau_s is None:
            # A lumped node with no time constant is not a model of anything: it cannot be
            # integrated, and silently defaulting the lag would put a number nobody authored
            # into the drift the whole M9 chain is built on. Same rule as the FPA node.
            raise ValueError("housing_temp_mode 'coupled' requires housing_tau_s (M9.3)")
        if self.vignetting_cos4 and self.distortion.model not in RECTILINEAR_MODELS:
            raise ValueError(
                f"vignetting_cos4 is a rectilinear-lens result and cannot be combined with the "
                f"{self.distortion.model!r} distortion model; supply vignetting_map instead "
                "(ADR 0015)"
            )
        return self


class DarkCurrentSpec(_Frozen):
    """§9.1 Arrhenius dark current: i(T) = i_ref (T/T_ref)^{3/2} exp(−E_g/2k (1/T − 1/T_ref))."""

    i_ref_a_per_pixel: float = Field(ge=0)
    t_ref_k: float = Field(gt=0)
    band_gap_ev: float = Field(gt=0)


class _FpaCommon(_Frozen):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pitch_um: float = Field(gt=0)
    fill_factor: float = Field(gt=0, le=1)
    frame_rate_hz: float = Field(gt=0)
    bit_depth: int = Field(ge=8, le=16)
    # FPA thermal node (§9.2 "FPA temperature coupling"), distinct from the optics housing node.
    # fpa_temp_mode is None while the node is not modelled; M9.2 builds it when it is set.
    fpa_temp_mode: FpaTempMode | None = Field(default=None)
    fpa_temp_k: float | None = Field(default=None, gt=0)
    fpa_tau_s: float | None = Field(default=None, gt=0)
    fpa_self_heating_k: float = Field(default=0.0, ge=0)
    # Raw, uncorrected response drift (M9.2, ADR 0053): gain(T) = 1 + sum c_i (T - T_cal)^i and
    # offset(T) = sum d_i (T - T_cal)^i, ascending powers from i = 1 so both are normalised at
    # T_cal by construction. What SURVIVES correction is nuc.residual_* -- do not conflate them.
    fpa_t_cal_k: float | None = Field(default=None, gt=0)
    fpa_gain_coeffs_per_k: tuple[float, ...] = ()
    fpa_offset_coeffs_dn_per_k: tuple[float, ...] = ()
    #: The **intrascene ceiling** of the gain state this camera is running in, kelvin (`PH.8`,
    #: ADR 0116). A Boson's two are 140 °C (high gain) and 500 °C (low gain);
    #: :data:`irsim.detector.gain_state.BOSON_GAIN_CEILING_K` has them. ``None`` is a core with no
    #: ceiling modelled, which is every camera in this repository before `PH.8` and is what keeps
    #: their goldens describing the pipeline. Applied to the at-aperture **radiance** plane --
    #: never to a temperature, which is a different operation once a pixel is a scene and not a
    #: blackbody.
    gain_ceiling_k: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _fpa_node(self) -> _FpaCommon:
        if self.fpa_temp_mode == "fixed" and self.fpa_temp_k is None:
            raise ValueError("fpa_temp_mode 'fixed' (TEC-pinned) needs fpa_temp_k")
        if self.fpa_temp_mode == "coupled" and self.fpa_tau_s is None:
            raise ValueError("fpa_temp_mode 'coupled' needs fpa_tau_s")
        if (self.fpa_gain_coeffs_per_k or self.fpa_offset_coeffs_dn_per_k) and self.t_cal_k is None:
            raise ValueError(
                "gain/offset(T_FPA) coefficients need a calibration temperature: set fpa_t_cal_k, "
                "or fpa_temp_k for a TEC-pinned core"
            )
        return self

    @property
    def t_cal_k(self) -> float | None:
        """The temperature the gain/offset polynomials are normalised at: the explicit
        ``fpa_t_cal_k`` if given, otherwise a TEC-pinned core's own set point."""
        return self.fpa_t_cal_k if self.fpa_t_cal_k is not None else self.fpa_temp_k


class BolometerFpa(_FpaCommon):
    """§12.2 ``fpa`` with ``type: bolometer`` (§9.2). Photon fields may be present only as
    ``null`` (the spec example writes them that way) -- a value is an error."""

    type: Literal["bolometer"]
    thermal_time_constant_ms: float = Field(gt=0)
    tcr_per_k: float
    # §9.2 membrane and readout constants the §12.2 block omits (ADR 0017); typical VOx values.
    absorptance: float = Field(default=0.8, gt=0, le=1)
    g_th_w_per_k: float = Field(default=1e-7, gt=0)
    bias_current_a: float = Field(default=50e-6, gt=0)
    resistance_ohm: float = Field(default=1e5, gt=0)
    quantum_efficiency: None = None
    well_capacity_e: None = None
    integration_time_ms: None = None
    dark_current_model: None = None

    @model_validator(mode="after")
    def _tcr(self) -> BolometerFpa:
        if self.tcr_per_k == 0:
            raise ValueError("tcr_per_k must be non-zero (VOx/a-Si are about -0.02 /K)")
        return self


class PhotonFpa(_FpaCommon):
    """§12.2 ``fpa`` with ``type: photon`` (§9.1). Bolometer fields may be present only as
    ``null``."""

    type: Literal["photon"]
    quantum_efficiency: float = Field(gt=0, le=1)
    well_capacity_e: float = Field(gt=0)
    integration_time_ms: float = Field(gt=0)
    dark_current_model: str
    # §9.1 parameters the §12.2 block omits (ADR 0017): required by the noise stage (M4).
    read_noise_e: float | None = Field(default=None, ge=0)
    dark_current: DarkCurrentSpec | None = None
    thermal_time_constant_ms: None = None
    tcr_per_k: None = None

    @model_validator(mode="after")
    def _integration_fits_frame(self) -> PhotonFpa:
        if self.integration_time_ms > 1000.0 / self.frame_rate_hz:
            raise ValueError(
                f"integration_time_ms {self.integration_time_ms} exceeds the frame period "
                f"{1000.0 / self.frame_rate_hz:.3f} ms"
            )
        return self


FpaSpec = Annotated[BolometerFpa | PhotonFpa, Field(discriminator="type")]


# Canonical order of the seven NVESD components (§10.2) wherever they travel as a vector.
RATIO_ORDER: tuple[str, ...] = ("t", "v", "h", "tv", "th", "vh", "tvh")


class Ratios3D(_Frozen):
    """§10.2 NVESD components relative to σ_TVH; ``tvh`` is the unit and must be exactly 1."""

    tvh: float = Field(ge=0)
    vh: float = Field(ge=0)
    h: float = Field(ge=0)
    v: float = Field(ge=0)
    tv: float = Field(ge=0)
    th: float = Field(ge=0)
    t: float = Field(ge=0)

    @model_validator(mode="after")
    def _unit(self) -> Ratios3D:
        if self.tvh != 1.0:
            raise ValueError("ratios_3d.tvh is the reference and must be exactly 1.0")
        return self

    def as_vector(self) -> tuple[float, ...]:
        """The seven ratios in ``RATIO_ORDER`` (t, v, h, tv, th, vh, tvh)."""
        return tuple(float(getattr(self, k)) for k in RATIO_ORDER)

    def total_over_tvh(self) -> float:
        """σ_total / σ_TVH = √Σ r² (variance closure, §10.2): 1.0604 for the Boson ratios."""
        return math.sqrt(sum(r * r for r in self.as_vector()))


class BadPixelTypeMix(_Frozen):
    """§10.4 defect classes as fractions of the bad-pixel population; must sum to 1."""

    dead: float = Field(default=0.4, ge=0, le=1)
    hot: float = Field(default=0.3, ge=0, le=1)
    flickering: float = Field(default=0.2, ge=0, le=1)
    blinking: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def _sums_to_one(self) -> BadPixelTypeMix:
        total = self.dead + self.hot + self.flickering + self.blinking
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"bad_pixel_type_mix must sum to 1, got {total:.6f}")
        return self


class NoiseSpec(_Frozen):
    """§12.2 ``noise``; ``netd_mk_at_300k`` anchors everything (§9.4).

    ``netd_ref_f_number`` is the working f-number the datasheet NETD was measured at (ADR 0025);
    ``None`` means the configured ``optics.f_number``. ``bad_pixel_type_mix`` splits the
    bad-pixel population by §10.4 class (ADR 0017 schema-gap policy).
    """

    netd_mk_at_300k: float = Field(gt=0)
    ratios_3d: Ratios3D
    fpn_drift_tau_s: float = Field(gt=0)
    bad_pixel_fraction: float = Field(ge=0, le=0.01)
    bad_pixel_cluster_lambda: float = Field(ge=0)
    netd_ref_f_number: float | None = Field(default=None, gt=0)
    bad_pixel_type_mix: BadPixelTypeMix = Field(default_factory=BadPixelTypeMix)
    # §10.4 random telegraph parameters for the flickering and blinking classes (M9.5a, ADR 0055).
    # The two-state chain is pinned by its stationary occupancy and its mean dwell in the bad
    # state; both dwell times are then geometric, which is what makes RTS RTS rather than a noisy
    # pixel. Occupancy is strictly below 1 because a pixel that never recovers is a dead one.
    bad_pixel_rts_occupancy: float = Field(default=0.3, gt=0, lt=1)
    bad_pixel_rts_dwell_frames: float = Field(default=8.0, gt=1)
    bad_pixel_rts_amplitude_dn: float = Field(default=400.0, ge=0)

    def sigma_ratios(self) -> tuple[float, ...]:
        return self.ratios_3d.as_vector()

    def total_over_tvh(self) -> float:
        return self.ratios_3d.total_over_tvh()


class NucSpec(_Frozen):
    """§12.2 ``nuc`` (§11.2)."""

    mode: NucMode
    ffc_interval_s: float = Field(gt=0)
    ffc_freeze_ms: float = Field(ge=0)
    residual_gain_ppm_per_k: float = Field(ge=0)
    residual_offset_mk_per_k: float = Field(ge=0)
    # Scene-based-correction time constant for `mode: shutterless` (M9.7, ADR 0057). A shutterless
    # core estimates the pattern from scene motion [R32, R33], so its residual settles where the
    # estimator's convergence balances the drift rather than growing without bound: a first-order
    # lag of this time constant. Unused in the other two modes.
    shutterless_tau_s: float = Field(default=120.0, gt=0)


class IspSpec(_Frozen):
    """§12.2 ``isp`` (§11.3, §11.4)."""

    agc: AgcMode
    plateau: float = Field(gt=0, lt=1)
    clip_percentiles: tuple[float, float]
    gamma: float = Field(gt=0)
    dde_gain: float = Field(ge=0)
    polarity: Polarity
    palette: Palette
    # Tiling for `agc: plateau_local` (M9.10); ignored by the global modes. (1, 1) is the global
    # operator exactly, which is the identity the local path is tested against.
    agc_tiles: tuple[int, int] = (8, 8)

    @model_validator(mode="after")
    def _clip(self) -> IspSpec:
        lo, hi = self.clip_percentiles
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(
                f"clip_percentiles must be ordered fractions in [0, 1], got {lo}, {hi}"
            )
        if min(self.agc_tiles) < 1:
            raise ValueError(f"agc_tiles must be at least 1 in each axis, got {self.agc_tiles}")
        return self


class FidelitySpec(_Frozen):
    """§15 Tier 5 caution 1: the ablation switches, as config rather than as arguments.

    §15 asks which parts of the model the sim-to-real gap actually depends on, and the answer is an
    ablation: run the same scenario with one mechanism off and compare. Every one of these already
    *existed* -- as a keyword argument of :meth:`irsim.pipeline.core.PipelineConfig.from_sensor` or
    of :func:`irsim.pipeline.sensor_chain.attach_sensor_chain`. **A keyword argument is invisible to
    the config hash**, so two ablation variants produced identical hashes, and a run's provenance
    could not say which of them it was. That is the gap this block closes; it adds no new physics.

    The other ablations §15 names were already explicit, hashed config and are deliberately *not*
    duplicated here, because a second way to set them would raise the question of which one wins:

    * **AGC** -- ``isp.agc: linear`` (or ``none``) is the ablation, and it is already in the hash.
    * **FFC** -- ``nuc.mode: ideal`` is a core that never shutters (``shutterless`` is the third
      real mode, not an ablation of the second).
    * **Clouds** -- cloud *fraction* is weather (``WeatherSeries.cloud_fraction``), never authored,
      so the cloud-free run is a weather file, and ``environment.clouds.tau`` sets what a cloud is.

    The default is full fidelity, which is what every configuration written before this block
    described. :data:`FULL_FIDELITY` is that default, and a config equal to it is **dropped from
    the config hash** -- so the hash tracks the ablation and not the notation, and a v8 file and a
    v9 file that spells out every switch as ``true`` hash the same.
    """

    #: Detector and 3-D noise (ADR 0022/0025). ``false`` is the ideal chain.
    noise: bool = True
    #: The optical PSF at the supersampled pitch (ADR 0059). ``false`` removes the optical MTF;
    #: the detector's own box MTF is geometry and stays.
    optical_psf: bool = True
    #: `OC.4`. The defocus term, independently of which model `optics.mtf.defocus_model` names --
    #: `false` renders the same camera perfectly focused, which is the ablation that measures what
    #: focus is worth. It cannot switch defocus *on*: that is the model's job.
    defocus: bool = True
    #: §10.4 dead/hot/flickering pixels and their replacement (M9 chain).
    bad_pixels: bool = True
    #: §11.2's post-FFC residual gain and offset drift (ADR 0056).
    nuc_residual: bool = True


#: The all-on default: the camera every configuration written before ME.8 describes.
FULL_FIDELITY = FidelitySpec()


class OutputsSpec(_Frozen):
    radiance_linear: bool
    apparent_temperature: bool
    dn_16: bool
    display_8: bool


class SensorSpec(_Frozen):
    """The ``sensor:`` block, with the derived read-only quantities the kernels need."""

    name: str = Field(min_length=1)
    band: BandSpec
    optics: OpticsSpec
    fpa: FpaSpec
    noise: NoiseSpec
    nuc: NucSpec
    isp: IspSpec
    outputs: OutputsSpec
    fidelity: FidelitySpec = FULL_FIDELITY

    @property
    def quantity(self) -> Literal["lb", "lb_q"]:
        """Which LUT table this camera runs on: ``"lb"`` (energy) or ``"lb_q"`` (photons).

        ADR 0021's rule, in one place. A bolometer measures absorbed *power*, so its transfer is
        linear in energy-form band radiance; a photon detector counts *photons*, so N_e =
        eta t_int Phi_q and the chain has to run on the photon table. The two differ by about
        1e19, and getting it wrong produces a uniformly, invisibly mis-scaled scene that the AGC
        then normalises away -- so every consumer asks the sensor rather than deciding for itself.
        """
        return "lb" if self.fpa.type == "bolometer" else "lb_q"

    @property
    def pixel_area_m2(self) -> float:
        """Photosensitive area A_d = pitch² · fill_factor, in m² (§9.1)."""
        return (self.fpa.pitch_um * 1e-6) ** 2 * self.fpa.fill_factor

    @property
    def nyquist_cyc_per_mm(self) -> float:
        """Detector Nyquist frequency 1 / (2 · pitch), cycles per mm (§8.3)."""
        return 1000.0 / (2.0 * self.fpa.pitch_um)

    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view of a pinhole with this focal length, degrees."""
        half_width_mm = self.fpa.width * self.fpa.pitch_um * 1e-3 / 2.0
        return math.degrees(2.0 * math.atan(half_width_mm / self.optics.focal_length_mm))

    @property
    def frame_period_s(self) -> float:
        return 1.0 / self.fpa.frame_rate_hz

    @property
    def dn_max(self) -> int:
        return int(2**self.fpa.bit_depth - 1)

    @property
    def fpa_shape(self) -> tuple[int, int]:
        """(height, width) of the native detector grid."""
        return (self.fpa.height, self.fpa.width)

    @property
    def detector_active_area_m2(self) -> float:
        """Alias of :attr:`pixel_area_m2` under the §9.1 name A_d."""
        return self.pixel_area_m2

    @property
    def active_width_um(self) -> float:
        """Side of the square active area, √fill_factor · pitch (the box width of MTF_det, §8.3)."""
        return math.sqrt(self.fpa.fill_factor) * self.fpa.pitch_um

    @model_validator(mode="after")
    def _focus_consistency(self) -> SensorSpec:
        """A focus distance with no defocus model would change nothing, silently.

        The reverse is fine and meaningful: `infinity` with a model is a lens focused past its
        hyperfocal, which still defocuses everything close. Only the combination that *looks* like
        it does something and does not is refused.
        """
        if self.optics.focus.mode != "infinity" and self.optics.mtf.defocus_model == "none":
            raise ValueError(
                f"optics.focus.mode is {self.optics.focus.mode!r} but optics.mtf.defocus_model is "
                "'none', so the focus distance would change nothing; name a model or drop the focus"
            )
        return self

    @property
    def focus_distance_m(self) -> float | None:
        """Where this camera is focused, in metres, or ``None`` for infinity (`OC.4`).

        ``hyperfocal`` is resolved here rather than at the call site because it depends on the
        pitch, which lives in a different block: H = f²/(F c) + f with c defaulting to one detector
        pitch. Resolving it once means the acceptable circle of confusion is stated in the config
        and nowhere else.
        """
        from irsim.optics.defocus import hyperfocal_distance_m

        focus = self.optics.focus
        if focus.mode == "infinity":
            return None
        if focus.mode == "fixed":
            return focus.distance_m
        coc_um = focus.coc_um if focus.coc_um is not None else self.fpa.pitch_um
        return hyperfocal_distance_m(self.optics.focal_length_mm, self.optics.f_number, coc_um)

    @property
    def defocus_enabled(self) -> bool:
        """Both switches have to agree: a model must be named **and** fidelity must allow it."""
        return self.optics.mtf.defocus_model != "none" and self.fidelity.defocus

    @property
    def reference_wavelength_um(self) -> float:
        """λ for the diffraction cut-off: ``mtf.reference_wavelength_um`` or the band centre."""
        if self.optics.mtf.reference_wavelength_um is not None:
            return self.optics.mtf.reference_wavelength_um
        return 0.5 * (self.band.lambda_min_um + self.band.lambda_max_um)

    @property
    def cutoff_cyc_per_mm(self) -> float:
        """Diffraction cut-off ξ_c = 1 / (λ F) in cycles per mm (§8.3), λ in mm."""
        return 1.0 / (self.reference_wavelength_um * 1e-3 * self.optics.f_number)


class SensorConfig(_Frozen):
    """A whole sensor file: ``schema_version`` (default 1) and the ``sensor:`` block."""

    schema_version: int = SCHEMA_VERSION
    sensor: SensorSpec

    @model_validator(mode="after")
    def _version(self) -> SensorConfig:
        if not MIN_SCHEMA_VERSION <= self.schema_version <= SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} outside the readable range "
                f"{MIN_SCHEMA_VERSION}-{SCHEMA_VERSION}"
            )
        return self
