"""Physical constants for IR radiometry.

Never define these inline elsewhere. If you need one that is not here, add it here with a source.

Unit convention (see the ir-radiometry skill):
  wavelength      always micrometres
  spectral radiance   W m^-2 sr^-1 um^-1
  band radiance       W m^-2 sr^-1
  temperature         kelvin

Sources: CODATA 2018 exact SI definitions for h, c, k_B, q.
docs/physics-model.md §16.3
"""

from typing import Final

# --- SI base ---------------------------------------------------------------
H_PLANCK: Final[float] = 6.62607015e-34  # J s      (exact)
C_LIGHT: Final[float] = 2.99792458e8  # m s^-1   (exact)
K_BOLTZMANN: Final[float] = 1.380649e-23  # J K^-1   (exact)
Q_E: Final[float] = 1.602176634e-19  # C        (exact)

# --- Derived, in the project's practical units ------------------------------
# Planck first radiation constant for spectral RADIANCE (energy form):
#   L(lam, T) = C1L / (lam**5 * (exp(C2 / (lam * T)) - 1))
# units: W um^4 m^-2 sr^-1   with lam in um  ->  L in W m^-2 sr^-1 um^-1
C1L: Final[float] = 1.1910429723971884e8

# Photon form: L_q(lam, T) = C1Q / (lam**4 * (exp(C2 / (lam * T)) - 1))
# units: photons s^-1 um^3 m^-2 sr^-1  ->  L_q in ph s^-1 m^-2 sr^-1 um^-1
C1Q: Final[float] = 5.995849160000001e26

# Second radiation constant, h c / k_B, in um K
C2: Final[float] = 1.4387768775039337e4

# Stefan-Boltzmann, W m^-2 K^-4
SIGMA_SB: Final[float] = 5.670374419e-8

# Wien displacement constant, um K
WIEN_B: Final[float] = 2897.771955

# Apéry's constant zeta(3) = sum 1/n^3: the photon-form analogue of pi^4/15.
ZETA_3: Final[float] = 1.2020569031595942

# Photon-flux Stefan-Boltzmann constant, photons s^-1 m^-2 K^-3:
#   M_q = SIGMA_Q * T^3  (hemispherical photon exitance of a blackbody)
# Derivation: integrate L_q = (2c/lam^4) / (exp(hc/(lam k T)) - 1) over lam, substituting
# t = hc/(lam k T):
#   int L_q dlam = 2c (kT/hc)^3 int_0^inf t^2/(e^t - 1) dt = 2c (kT/hc)^3 * 2 zeta(3).
# Exitance is pi times radiance, so SIGMA_Q = 4 pi zeta(3) k^3 / (h^3 c^2) = 1.5205e15.
SIGMA_Q: Final[float] = (
    4.0 * 3.141592653589793 * ZETA_3 * K_BOLTZMANN**3 / (H_PLANCK**3 * C_LIGHT**2)
)

# --- Atmosphere ------------------------------------------------------------
# Specific gas constant of water vapour, J kg^-1 K^-1 (R / M_water = 8.314462618 / 0.018015):
# absolute humidity rho_v = e / (R_v T); the §7.3 factor 216.7 g m^-3 hPa^-1 K is 1e5 / R_v.
R_V_WATER: Final[float] = 461.5
# Koschmieder constant: meteorological optical range V is the path over which the contrast of a
# black target against the horizon sky falls to the 2 % threshold, V = ln(1/0.02) / gamma_vis
# (Koschmieder 1924; WMO Guide to Meteorological Instruments, MOR definition). Kept as the
# exact ln 50 so tau(V) = 0.02 holds to round-off; the conventional rounded value is 3.912.
KOSCHMIEDER: Final[float] = 3.912023005428146  # ln(50)

# --- Solar --------------------------------------------------------------------
# Total solar irradiance at 1 AU, W m^-2 (Kopp & Lean 2011, SORCE/TIM: 1360.8 +- 0.5). Upper bound
# for any surface irradiance in a weather file; the solar-path transmittance (M11) scales it.
SOLAR_CONSTANT_W_M2: Final[float] = 1361.0

# --- Moist air and evaporation (docs/physics-model.md §6.1 with the latent term; PH.1) ------------
# Latent heat of vaporisation of water at 20 C, J kg^-1 (2.501e6 at 0 C, 2.45e6 at 20 C: Rogers &
# Yau, A Short Course in Cloud Physics, Table 2.1). The roadmap's 2.45 MJ/kg is this value.
L_V_WATER_J_KG: Final[float] = 2.45e6
# Latent heat of fusion of water at 0 C, J kg^-1 (Rogers & Yau, Table 2.1: 3.34e5). A melting
# snow surface is held at 273.15 K and every surplus joule goes here instead of into temperature,
# which is why a snowfield reads flat at exactly freezing on a sunny spring afternoon (PH.10).
L_F_WATER_J_KG: Final[float] = 3.34e5
# The melting point of water at one atmosphere, K. The triple point is 273.16 K; the 0.01 K
# difference is far below anything this project resolves, and 273.15 is what a snow model caps at.
T_MELT_WATER_K: Final[float] = 273.15
# Specific heat of dry air at constant pressure, J kg^-1 K^-1, and a standard-atmosphere density
# at sea level, 15 C (ICAO): the Lewis relation h = rho c_p C_H U ties the sensible and latent
# bulk fluxes together, and the psychrometric wet bulb is where they balance.
C_P_AIR_J_KGK: Final[float] = 1005.0
RHO_AIR_STD_KG_M3: Final[float] = 1.225
P_STD_HPA: Final[float] = 1013.25
# Standard gravity (CGPM 1901) and the ICAO sea-level temperature the standard density is quoted
# at. Buoyancy-driven correlations -- Heskestad's fire plume (PH.7) -- need both: the first in the
# group that sets the plume's strength, the second to put the ambient density on the scene's own
# air temperature instead of on 15 C.
G_STANDARD_M_S2: Final[float] = 9.80665
T_STD_ICAO_K: Final[float] = 288.15
# Ratio of the molar masses of water vapour and dry air: q = eps e / (p - (1 - eps) e).
EPSILON_WATER_AIR: Final[float] = 0.622
# Bulk transfer coefficient for latent heat (Dalton number) at 10 m in the 5-10 m/s range, COARE
# 3.6 (Fairall et al. 2003 / Edson et al. 2013): 1.1-1.2e-3. Held constant here; COARE's own
# stability and gustiness dependence is what PH.1's 15 % test tolerance allows for. ESTIMATED.
C_E_BULK: Final[float] = 1.15e-3

# --- Soot (docs/physics-model.md §2 read for a participating medium; PH.4) -----------------------
# Small-particle (Rayleigh) soot absorbs as kappa_lambda = C0 * f_v / lambda, with f_v the soot
# volume fraction and C0 = 36 pi n k / ((n^2 - k^2 + 2)^2 + 4 n^2 k^2) from the soot refractive
# index. Reported C0 spans 4.9-7.9 across measured optical constants; 7.0 is Widmann's recommended
# value (Widmann, Combust. Sci. Tech. 175 (2003) 2299: "Evaluation of the Planck mean absorption
# coefficients for radiation from soot") and the value FDS/RadCal use for kappa_P = 3.72 C0 f_v
# T / C2. ESTIMATED to that spread, not measured for any particular flame.
SOOT_RAYLEIGH_C0: Final[float] = 7.0

# --- Semiconductor band gaps for the Arrhenius dark-current model (docs/physics-model.md §9.1) ----
# i_dark ∝ T^1.5 exp(-E_g / 2 k_B T). Values at the detectors' operating temperatures:
#   InSb   0.23 eV at 77 K   (Littler & Seiler 1985; Vurgaftman et al. 2001 -- 0.235 eV at 0 K)
#   InGaAs 0.75 eV at 300 K  (In0.53Ga0.47As lattice-matched to InP; Vurgaftman et al. 2001)
BAND_GAP_INSB_EV: Final[float] = 0.23
BAND_GAP_INGAAS_EV: Final[float] = 0.75
EV_PER_K: Final[float] = 8.617333262e-5  # k_B in eV/K (CODATA 2018)

# --- Temperature encoding and LUT grid --------------------------------------
# One definition, shared by the config layer, the LUT builder, the Warp/SPG kernels and the
# Isaac AOV adapter. The G-buffer carries c = (T - T_ENCODE_REF_K) / T_ENCODE_SPAN_K in float32,
# never raw kelvin through fp16 (docs/physics-model.md §13.3, §3.2 precision trap, CLAUDE.md #2).
T_ENCODE_REF_K: Final[float] = 200.0
T_ENCODE_SPAN_K: Final[float] = 800.0

# Band LUT grid (docs/physics-model.md §3.2 b, §13.5): T in [200, 1000] K at 0.05 K, 16001 entries.
LUT_T_MIN_K: Final[float] = 200.0
LUT_T_MAX_K: Final[float] = 1000.0
LUT_DT_K: Final[float] = 0.05
LUT_N: Final[int] = 16001  # round((LUT_T_MAX_K - LUT_T_MIN_K) / LUT_DT_K) + 1

# --- Numerical guards -------------------------------------------------------
# exp() overflows for x beyond ~709 in float64. Cold scenes at short wavelengths
# reach this easily (200 K at 0.8 um gives x ~ 90, but 100 K at 0.4 um gives ~360,
# and users will try worse). Clip and return ~0 radiance rather than inf/nan.
EXP_ARG_MAX: Final[float] = 700.0

# Plausibility bounds asserted at API boundaries.
WAVELENGTH_MIN_UM: Final[float] = 0.1
WAVELENGTH_MAX_UM: Final[float] = 1000.0
TEMPERATURE_MIN_K: Final[float] = 1.0
TEMPERATURE_MAX_K: Final[float] = 6000.0

# --- Dry-air gas properties -------------------------------------------------
# Needed by the aerodynamic-heating model of an airborne skin (§6.6 by extension; ADR 0075):
# the speed of sound sets the Mach number, and gamma and Pr set the recovery factor.
# Sources: specific gas constant R = R_universal / M_air = 8.314462618 / 0.0289647 (CIPM 2007
# standard dry-air molar mass); gamma and Pr are the textbook near-room-temperature dry-air
# values (e.g. White, *Viscous Fluid Flow*), both weak functions of temperature that are treated
# as constants over the 200-320 K range this model is used in.
R_SPECIFIC_AIR: Final[float] = 287.0528  # J kg^-1 K^-1
GAMMA_AIR: Final[float] = 1.4  # ratio of specific heats, dry air
PRANDTL_AIR: Final[float] = 0.71

# --- Seawater and near-surface air thermophysical properties ----------------
# Needed by the sea skin-temperature model (MM.4, ADR 0080): the cool-skin deficit is a molecular
# conduction problem across a sub-millimetre sublayer, so it is set by the kinematic viscosity and
# the thermal conductivity of the water, and the sublayer thickness is set by the wind stress
# delivered through the air.
#
# Sources and validity. Seawater values are for 35 PSU at ~20 degC, the conditions the cool-skin
# literature is written for (Saunders 1967; Fairall et al. 1996, JGR 101, 1295): the kinematic
# viscosity from the Sharqawy et al. (2010) correlation, the thermal conductivity from the same
# review, the density likewise. All three vary by a few per cent over 0-30 degC, which is small
# beside the factor-of-two spread in the Saunders proportionality constant itself, so they are
# treated as constants and the error is recorded in ADR 0080 rather than carried as a temperature
# dependence nothing else in the model would deserve.
#
# The broadband emissivity is the thermal-infrared hemispherical value for seawater, used ONLY in
# the broadband energy balance that drives the cool skin. It is deliberately not the band-effective
# emissivity of `configs/materials/water.yaml`, which is what the camera sees in one band at one
# angle: an energy balance integrates over all wavelengths and the whole hemisphere, and using a
# band value there would be a different physical quantity wearing the same name.
KINEMATIC_VISCOSITY_SEAWATER: Final[float] = 1.05e-6  # m^2 s^-1, 35 PSU at 20 degC
THERMAL_CONDUCTIVITY_SEAWATER: Final[float] = 0.596  # W m^-1 K^-1
DENSITY_SEAWATER: Final[float] = 1025.0  # kg m^-3
DENSITY_AIR_SEA_LEVEL: Final[float] = 1.225  # kg m^-3, ISA at 15 degC
EMISSIVITY_SEAWATER_BROADBAND: Final[float] = 0.98

# --- The sun's angular size -------------------------------------------------
# The photosphere subtends a finite disc, which is why a shadow edge is a ramp and not a step
# (PT.22, ADR 0107). The mean apparent diameter is 31.99 arcmin = 0.5332 deg, varying between
# 31.46' at aphelion and 32.53' at perihelion (Astronomical Almanac 2024, section C); the 1.7 %
# annual swing is well inside the 10 % the penumbra-width test allows, so the mean is used and
# the eccentricity is not carried. Half of it is the half-angle a ray may depart the sun
# direction by and still come from the disc.
SOLAR_DISC_DIAMETER_DEG: Final[float] = 0.5332

# --- Fresh water ------------------------------------------------------------
# The pond/lake/puddle counterparts of the seawater values above, for the still-water skin model
# (PH.3, ADR 0108). Fresh water at 20 degC: the kinematic viscosity and thermal conductivity from
# the IAPWS-95 formulation as tabulated in the CRC Handbook (97th ed.), the density and specific
# heat likewise. They differ from the seawater figures by 4 %, 0.3 % and 2.6 % respectively --
# small, and kept separate anyway because a puddle is not a sea and a reader should not have to
# discover that the model used salt water for a road.
KINEMATIC_VISCOSITY_FRESH_WATER: Final[float] = 1.004e-6  # m^2 s^-1, 20 degC
THERMAL_CONDUCTIVITY_FRESH_WATER: Final[float] = 0.598  # W m^-1 K^-1
DENSITY_FRESH_WATER: Final[float] = 998.2  # kg m^-3
SPECIFIC_HEAT_FRESH_WATER: Final[float] = 4182.0  # J kg^-1 K^-1
